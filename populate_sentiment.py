import argparse
import os
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
    # parse script params
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--episode_keys", "-e", help="Episode keys", required=False)
    parser.add_argument("--season", "-n", help="Season", required=False)
    parser.add_argument("--analyzer", "-a", help="Analyzer", required=True)
    parser.add_argument("--scene_level", "-c", help="Scene level", required=False)
    parser.add_argument("--line_level", "-l", help="Line level", required=False)
    parser.add_argument("--overwrite_csv", "-o", help="Overwrite CSV file", required=False)
    parser.add_argument("--write_to_es", "-w", help="Write to es", required=False)
    args = parser.parse_args()
    # assign script params to vars
    episode_keys = []
    season = None
    scene_level = False
    line_level = False
    overwrite_csv = False
    write_to_es = False
    if args.episode_keys: 
        episode_keys = args.episode_keys
    if args.season: 
        season = args.season
    if args.scene_level: 
        scene_level = args.scene_level
    if args.line_level: 
        line_level = args.line_level
    if args.overwrite_csv: 
        overwrite_csv = args.overwrite_csv
    if args.write_to_es: 
        write_to_es = args.write_to_es

    if episode_keys:
        e_keys = episode_keys.split(',')
        for e_key in e_keys:
            populate_episode_sentiment(args.show_key, e_key, args.analyzer, scene_level=scene_level, line_level=line_level, overwrite_csv=overwrite_csv, write_to_es=write_to_es)
    elif season:
        simple_episodes_response = esr.fetch_simple_episodes(ShowKey(args.show_key), season=season)
        simple_episodes = simple_episodes_response['episodes']
        for ep in simple_episodes:
            populate_episode_sentiment(args.show_key, ep['episode_key'], args.analyzer, scene_level=scene_level, line_level=line_level, overwrite_csv=overwrite_csv, write_to_es=write_to_es)
    else:
        print(f'Either `episode_key` (-e) or `season` (-n) is required, populating sentiment for an entire series in a single job is currently not supported')
        return 


def populate_episode_sentiment(show_key: str, episode_key: str, analyzer: str, scene_level: bool = False, line_level: bool = False, overwrite_csv: bool = False, write_to_es: bool = False) -> tuple[pd.DataFrame, dict]:
    '''
    Generate and populate sentiment for episode. Currently populating to 3 places: 
    1. dict -> response
    2. dataframe -> csv file
    3. es index (optional)
    '''
    start_ts = datetime.datetime.now()
    print('----------------------------------------------------------------------------------------------------')
    print(f'begin populate_episode_sentiment episode_key={episode_key} analyzer={analyzer} scene_level={scene_level} line_level={line_level} overwrite_csv={overwrite_csv} write_to_es={write_to_es} at start_ts={str(start_ts)[:19]}')

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
    episode_sent_df['type'] = 'E'
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
    if scene_level or line_level:

        # scene-level processing will use fetch_flattened_scenes, trusting (gulp) that scene index positions align with their es_episode.scenes counterparts
        if scene_level:
            flattened_scenes_response = esr.fetch_flattened_scenes(ShowKey(show_key), episode_key)
            flattened_scenes = flattened_scenes_response['flattened_scenes']
            flattened_scenes_with_speakers_response = esr.fetch_flattened_scenes(ShowKey(show_key), episode_key, include_speakers=True, line_breaks=True)
            flattened_scenes_with_speakers = flattened_scenes_with_speakers_response['flattened_scenes']

        # both scene- and line-level processing iterate over es_episode.scenes, carefully tracking scene index position
        print(f'begin processing {len(es_episode.scenes)} scenes')
        for scene_i in range(len(es_episode.scenes)):
            es_scene = es_episode.scenes[scene_i]
            # scene-level: analyze flattened_scene
            if scene_level:
                print(f'executing populate_episode_sentiment on flattened_scene at scene_i={scene_i}')
                if not flattened_scenes[scene_i]:
                    print(f'flattened_scene at scene_i={scene_i} contains no dialog text, skipping')
                    continue

                # process flattened scene as one uninterrupted blob of text
                total_reqs += 1
                scene_sent_df, scene_sent_dict = sa.generate_sentiment(flattened_scenes[scene_i], analyzer)
                if scene_sent_df is None:
                    failure_message = f'failure to execute generate_sentiment on flattened_scene at scene_i={scene_i} with text=`{flattened_scenes[scene_i]}`'
                    failure_reqs.append(failure_message)
                    print(failure_message)
                else:
                    success_reqs += 1
                    # add contextual properties to scene_sent_df, concat with episode_sent_df
                    scene_sent_df['key'] = f'S{scene_i}'
                    scene_sent_df['type'] = 'S'
                    scene_sent_df['scene'] = scene_i
                    scene_sent_df['line'] = 'ALL'
                    scene_sent_df['speaker'] = 'ALL'
                    episode_sent_df = pd.concat([episode_sent_df, scene_sent_df], axis=0, ignore_index=True)

                # process flattened scene by attributing different emotional sentiment to each speaker in the scene (at the scene-level, not the line-level)
                # requires special prompting so only applicable to openai
                if analyzer == 'openai_emo': 
                    total_reqs += 1
                    scene_sent_multi_speaker_df, scene_sent_multi_speaker_dict = sa.generate_sentiment(flattened_scenes_with_speakers[scene_i], analyzer, multi_speaker=True)
                    if scene_sent_multi_speaker_df is None:
                        # NOTE this should never happen as it would have been caught in previous loop against flattened_scenes
                        failure_message = f'failure to execute generate_sentiment on flattened_scenes_with_speakers at scene_i={scene_i} with text=`{flattened_scenes_with_speakers[scene_i]}`'
                        failure_reqs.append(failure_message)
                        print(failure_message)
                    else:
                        success_reqs += 1
                        # add contextual properties to scene_sent_df, concat with episode_sent_df
                        scene_sent_multi_speaker_df['key'] = f'S{scene_i}'
                        scene_sent_multi_speaker_df['type'] = 'SD'
                        scene_sent_multi_speaker_df['scene'] = scene_i
                        scene_sent_multi_speaker_df['line'] = 'ALL'
                        episode_sent_df = pd.concat([episode_sent_df, scene_sent_multi_speaker_df], axis=0, ignore_index=True)

                # update es object
                if write_to_es:
                    if analyzer == 'openai_emo':
                        for emo in OPENAI_EMOTIONS:
                            set_dict_value_as_es_value(es_scene, scene_sent_dict, emo, 'openai_sent_') 
                        es_scene.speaker_summaries = scene_sent_multi_speaker_dict
                    elif analyzer == 'nltk_pol':
                        for pol in NTLK_POLARITY:
                            set_dict_value_as_es_value(es_scene, scene_sent_dict, pol, 'nltk_sent_')

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
                        line_sent_df['key'] = f'S{scene_i}L{line_i}'
                        line_sent_df['type'] = 'L'
                        line_sent_df['scene'] = scene_i
                        line_sent_df['line'] = line_i
                        line_sent_df['speaker'] = es_scene_event.spoken_by
                        line_i += 1
                        episode_sent_df = pd.concat([episode_sent_df, line_sent_df], axis=0, ignore_index=True)
                        # update es object
                        if write_to_es:
                            if analyzer == 'openai_emo':
                                for emo in OPENAI_EMOTIONS:
                                    set_dict_value_as_es_value(es_scene_event, line_sent_dict, emo, 'openai_sent_')
                            elif analyzer == 'nltk_pol':
                                for pol in NTLK_POLARITY:
                                    set_dict_value_as_es_value(es_scene_event, line_sent_dict, pol, 'nltk_sent_')

    # write to es
    if write_to_es:
        esqb.save_es_episode(es_episode)

    end_ts = datetime.datetime.now()
    duration = end_ts - start_ts
    print('* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *')
    print(f'finish populate_episode_sentiment for episode {episode_key} in {duration.seconds} seconds at end_ts={str(end_ts)[:19]}')

    # use dataframe to upsert csv file
    file_path = f'sentiment_data/{show_key}/{analyzer}/{show_key}_{episode_key}.csv'
    # episode_sent_df.to_csv(file_path, sep=',', header=True)
    write_csv(file_path, episode_sent_df, scene_level=scene_level, line_level=line_level, overwrite=overwrite_csv)

    req_report = {
        'episode_key': episode_key,
        'analyzer': analyzer,
        'scene_level': scene_level,
        'line_level': line_level,
        'write_to_es': write_to_es,
        'total_reqs': total_reqs,
        'success_reqs': success_reqs,
        'failure_reqs': len(failure_reqs),
        'success_rate': f'{round(success_reqs / total_reqs) * 100}%',
        'duration': f'{duration.seconds} s',
        'avg_req_duration': f'{round(duration.seconds / total_reqs)} s',
    }

    print(f'req_report={req_report}')

    # return episode_sent_df, req_report


def write_csv(file_path: str, df: pd.DataFrame, scene_level: bool, line_level: bool, overwrite: bool = False):
    # if overwriting or previous file doesn't exist, simply write full df contents to file_path
    if overwrite or not os.path.isfile(file_path):
        df.to_csv(file_path, sep=',', header=True, index=False)
        return

    # if csv file already exists, load into prev_df and selectively concat/overwrite with df based on request granularity
    if os.path.isfile(file_path):
        prev_df = pd.read_csv(file_path, sep=',')
        # episode-level data is always generated, and always overwritten
        prev_df = prev_df.loc[prev_df['type'] != 'E']
        # scene-level data is only overwritten if it has been newly (re-)generated
        if scene_level:
            prev_df = prev_df.loc[~prev_df['type'].isin(['S', 'SD'])]
            # prev_df = prev_df.loc[prev_df['type'] != 'S']
            # prev_df = prev_df.loc[prev_df['type'] != 'SD']
        # live-level data is only overwritten if it has been newly (re-)generated
        if line_level:
            prev_df = prev_df.loc[prev_df['type'] != 'L']
        df = pd.concat([prev_df, df], axis=0, ignore_index=True)
        df.to_csv(file_path, sep=',', header=True, index=False)


if __name__ == '__main__':
    main()
