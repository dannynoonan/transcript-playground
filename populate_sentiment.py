import argparse
import pandas as pd
import datetime

from app.es.es_model import EsEpisodeTranscript, EsEpisodeNarrativeSequence, EsSpeaker, EsSpeakerSeason, EsSpeakerEpisode
import app.es.es_query_builder as esqb
import app.es.es_read_router as esr
import app.nlp.sentiment_analyzer as sa
from app.nlp.nlp_metadata import OPENAI_EMOTIONS, NTLK_POLARITY, SENTIMENT_ANALYZERS
from app.show_metadata import ShowKey
from app.utils import set_dict_value_as_es_value


def main():
    # start_ts = str(datetime.datetime.now())[:19]
    start_ts = datetime.datetime.now()
    print(f'begin populate_sentiment against full episode at start_ts={str(start_ts)[:19]}')
    # parse script params
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--episode_key", "-e", help="Episode key", required=True)
    parser.add_argument("--analyzer", "-a", help="Analyzer", required=True)
    parser.add_argument("--scene_level", "-c", help="Scene level", required=False)
    parser.add_argument("--line_level", "-l", help="Line level", required=False)
    parser.add_argument("--write_to_es", "-w", help="Write to es", required=False)
    args = parser.parse_args()
    # assign script params to vars
    scene_level = False
    line_level = False
    write_to_es = False
    if args.scene_level: 
        scene_level = args.scene_level
    if args.line_level: 
        line_level = args.line_level
    if args.write_to_es: 
        write_to_es = args.write_to_es
    episode_emo_df, req_report = populate_episode_sentiment(
        args.show_key, args.episode_key, args.analyzer, scene_level=scene_level, line_level=line_level, write_to_es=write_to_es)

    end_ts = datetime.datetime.now()
    duration = end_ts - start_ts
    req_report['duration'] = duration.seconds
    print(f'finish populate_episode_sentiment against full episode in {duration.seconds} seconds at end_ts={str(end_ts)[:19]}')

    # write dataframe to csv
    file_path = f'sentiment_data/{args.show_key}/{args.analyzer}/{args.show_key}_{args.episode_key}.csv'
    episode_emo_df.to_csv(file_path, sep=',', header=True)

    print(f'req_report={req_report}')


def populate_episode_sentiment(show_key: str, episode_key: str, analyzer: str, scene_level: bool = False, line_level: bool = False, write_to_es: bool = False) -> tuple[pd.DataFrame, dict]:
    '''
    Generate and populate sentiment for episode. Currently populating to 3 places: 
    1. dict -> response
    2. dataframe -> csv file
    3. es index (optional)
    '''
    if analyzer not in SENTIMENT_ANALYZERS:
        print(f'`{analyzer}` in not a valid sentiment analyzer, supported analyzers are {SENTIMENT_ANALYZERS}')
        return 

    try:
        es_episode = EsEpisodeTranscript.get(id=f'{show_key}_{episode_key}')
    except Exception as e:
        print(f'Failure to fetch es_episode matching show_key={show_key} episode_key={episode_key}: {e}')
        return

    total_reqs = 0
    success_reqs = 0
    failure_reqs = []

    # episode-level sentiment 
    total_reqs += 1
    episode_sent_df, episode_sent_dict = sa.generate_sentiment(es_episode.flattened_text, analyzer)
    if episode_sent_df is None:
        print(f"failure to execute populate_episode_sentiment at episode-level for show_key={show_key} episode_key={episode_key} analyzer={analyzer}")
        return
    success_reqs += 1
    # add contextual properties to episode_sent_df
    episode_sent_df['key'] = 'E'
    episode_sent_df['scene'] = 'ALL'
    episode_sent_df['line'] = 'ALL'
    episode_sent_df['speaker'] = 'ALL'
    # update es object
    if write_to_es:
        if analyzer == 'openai_emo':
            for emo in OPENAI_EMOTIONS:
                set_dict_value_as_es_value(es_episode, episode_sent_dict, emo, 'openai_sent_')
        elif analyzer == 'nltk_pol':
            for pol in NTLK_POLARITY:
                set_dict_value_as_es_value(es_episode, episode_sent_dict, pol, 'nltk_sent_')

    # scene- and line-level sentiment 
    # scene_sent_dicts = [] # TODO this isn't being used
    if scene_level or line_level:

        # scene-level processing will use fetch_flattened_scenes, trusting (gulp) that scene index positions align with their es_episode.scenes counterparts
        if scene_level:
            flattened_scenes_response = esr.fetch_flattened_scenes(ShowKey(show_key), episode_key)
            flattened_scenes = flattened_scenes_response['flattened_scenes']
            flattened_scenes_with_speakers_response = esr.fetch_flattened_scenes(ShowKey(show_key), episode_key, include_speakers=True, line_breaks=True)
            flattened_scenes_with_speakers = flattened_scenes_with_speakers_response['flattened_scenes']

        # both scene- and line-level processing iterate over es_episode.scenes, carefully tracking scene index position
        for scene_i in range(len(es_episode.scenes)):
            es_scene = es_episode.scenes[scene_i]
            # scene_sent_dict = dict(scene_i=scene_i, scene_level=None, multi_speaker=None, line_level=[])
            # scene_sent_dicts.append(scene_sent_dict) # TODO this isn't being used

            # scene-level: analyze flattened_scene
            if scene_level:
                print(f'executing populate_episode_sentiment on flattened_scene at scene_i={scene_i}')
                if not flattened_scenes[scene_i]:
                    print(f'flattened_scene at scene_i={scene_i} contains no dialog text, skipping')
                    continue

                # process flattened scene as one uninterrupted blob of text
                total_reqs += 1
                scene_sent_df, scene_level_dict = sa.generate_sentiment(flattened_scenes[scene_i], analyzer)
                # scene_sent_df, scene_sent_dict['scene_level'] = sa.generate_sentiment(flattened_scenes[scene_i], analyzer)
                if scene_sent_df is None:
                    failure_message = f'failure to execute generate_sentiment on flattened_scene at scene_i={scene_i} with text=`{flattened_scenes[scene_i]}`'
                    failure_reqs.append(failure_message)
                    print(failure_message)
                    continue
                success_reqs += 1
                # add contextual properties to scene_sent_df, concat with episode_sent_df
                scene_sent_df['key'] = f'S{scene_i}'
                scene_sent_df['scene'] = scene_i
                scene_sent_df['line'] = 'ALL'
                scene_sent_df['speaker'] = 'ALL'
                episode_sent_df = pd.concat([episode_sent_df, scene_sent_df], axis=0)

                # process flattened scene by attributing different emotional sentiment to each speaker in the scene (at the scene-level, not the line-level)
                # requires special prompting so only applicable to openai
                if analyzer == 'openai_emo': 
                    total_reqs += 1
                    # scene_sent_multi_speaker_df, scene_sent_dict['multi_speaker'] = sa.generate_sentiment(flattened_scenes_with_speakers[scene_i], analyzer, multi_speaker=True)
                    scene_sent_multi_speaker_df, multi_speaker_dict = sa.generate_sentiment(flattened_scenes_with_speakers[scene_i], analyzer, multi_speaker=True)
                    if scene_sent_multi_speaker_df is None:
                        # NOTE this should never happen as it would have been caught in previous loop against flattened_scenes
                        failure_message = f'failure to execute generate_sentiment on flattened_scenes_with_speakers at scene_i={scene_i} with text=`{flattened_scenes_with_speakers[scene_i]}`'
                        failure_reqs.append(failure_message)
                        print(failure_message)
                        print('******** NOTE: this should never happen, error should have been caught with flattened_scene (without speakers) *********')
                        continue
                    success_reqs += 1
                    # add contextual properties to scene_sent_df, concat with episode_sent_df
                    scene_sent_multi_speaker_df['key'] = f'S{scene_i}'
                    scene_sent_multi_speaker_df['scene'] = scene_i
                    scene_sent_multi_speaker_df['line'] = 'ALL'
                    episode_sent_df = pd.concat([episode_sent_df, scene_sent_multi_speaker_df], axis=0)

                # update es object
                if write_to_es:
                    if analyzer == 'openai_emo':
                        for emo in OPENAI_EMOTIONS:
                            set_dict_value_as_es_value(es_scene, scene_level_dict, emo, 'openai_sent_')
                        es_scene.speaker_summaries = multi_speaker_dict
                    elif analyzer == 'nltk_pol':
                        for pol in NTLK_POLARITY:
                            set_dict_value_as_es_value(es_scene, scene_level_dict, pol, 'nltk_sent_')

            # line-level: analyze dialog for each line in scene
            if line_level:
                line_i = 0
                for es_scene_event in es_scene.scene_events:
                    if es_scene_event.spoken_by and es_scene_event.dialog:
                        print(f'executing populate_episode_sentiment on flattened_scene at scene_i={scene_i} line_i={line_i}')
                        total_reqs += 1
                        line_sent_df, line_sent_dict = sa.generate_sentiment(es_scene_event.dialog, analyzer)
                        if line_sent_df is None:
                            failure_message = f'failure to execute populate_episode_sentiment on flattened_scene at scene_i={scene_i} line_i={line_i} es_scene_event.dialog=`{es_scene_event.dialog}`'
                            failure_reqs.append(failure_message)
                            print(failure_message)
                            continue
                        success_reqs += 1
                        # add contextual properties to line_sent_df, concat with episode_sent_df
                        # scene_sent_dict['line_level'].append(line_sent_dict)
                        line_sent_df['key'] = f'S{scene_i}L{line_i}'
                        line_sent_df['scene'] = scene_i
                        line_sent_df['line'] = line_i
                        line_sent_df['speaker'] = es_scene_event.spoken_by
                        line_i += 1
                        episode_sent_df = pd.concat([episode_sent_df, line_sent_df], axis=0)
                        # update es object
                        if write_to_es:
                            if analyzer == 'openai_emo':
                                for emo in OPENAI_EMOTIONS:
                                    set_dict_value_as_es_value(es_scene_event, line_sent_dict, emo, 'openai_sent_')
                            elif analyzer == 'nltk_pol':
                                for pol in NTLK_POLARITY:
                                    set_dict_value_as_es_value(es_scene_event, line_sent_dict, pol, 'nltk_sent_')

    req_report = {
        'openai_total_reqs': total_reqs,
        'openai_success_reqs': success_reqs,
        'openai_failure_reqs': failure_reqs,
    }

    # write to es
    if write_to_es:
        esqb.save_es_episode(es_episode)

    return episode_sent_df, req_report


if __name__ == '__main__':
    main()
