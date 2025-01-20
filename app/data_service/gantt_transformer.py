from operator import itemgetter
import pandas as pd

from app.app_metadata import GANTT_DATA_DIR
from app.auth import ADMIN_USER
import app.routers.es_read_router as esr
from app.show_metadata import ShowKey, show_metadata, EPISODE_TOPIC_GROUPINGS


def generate_episode_gantt_sequence(show_key: ShowKey, episode_key: str) -> tuple[list, list]:
    '''
    TODO
    '''

    max_line_chars = 280
    dialog_timeline = []
    location_timeline = []
    word_i = 0
    scene_start_i = 0
    # fetch episode data
    episode = esr.fetch_episode(show_key, episode_key, ADMIN_USER)
    es_episode = episode['es_episode']
    if 'scenes' not in es_episode:
        return {"dialog_timeline": [], "location_timeline": []}
    # for each scene containing dialog:
    #   - for each dialog scene_event, add a dialog_span specifying speaker and start/end word index of dialog
    #   - add a location_span specifying location and start/end word index of scene
    for i, s in enumerate(es_episode['scenes']):
        if 'scene_events' not in s:
            continue
        scene_lines = []
        for j, se in enumerate(s['scene_events']):
            if 'spoken_by' and 'dialog' in se:
                line_dialog = se['dialog']
                line_wc = len(line_dialog.split())
                if len(line_dialog) > max_line_chars:
                    line_dialog = f'{line_dialog[:max_line_chars]}...'
                dialog_span = dict(Task=se['spoken_by'], Start=word_i, Finish=(word_i+line_wc-1), Line=line_dialog, scene=i, scene_event=j)
                dialog_timeline.append(dialog_span)
                word_i += line_wc
                scene_lines.append(f"{se['spoken_by']}: {line_dialog}")
        location_span = dict(Task=s['location'], Start=scene_start_i, Finish=(word_i-1), Line='<br>'.join(scene_lines), scene=i)
        location_timeline.append(location_span)
        scene_start_i = word_i

    return dialog_timeline, location_timeline


def generate_series_speaker_gantt_sequence(show_key: ShowKey, limit_cast: bool = False, overwrite_file: bool = False) -> tuple[dict, list]:
    '''
    TODO
    '''

    episodes_to_speaker_line_counts = {}
    episode_speakers_sequence = []
    
    # get ordered list of all episodes
    response = esr.fetch_simple_episodes(show_key, ADMIN_USER)
    episodes = response['episodes']

    # for each episode:
    # - fetch all speakers ordered by scene_event count (how many lines they have)
    # - transform results into lists of span dicts for creating plotly gantt charts
    episode_i = 0
    for episode in episodes:
        episode_key = episode['episode_key']
        episode_title = episode['title']
        episode_season = episode['season']
        sequence_in_season = episode['sequence_in_season']

        # fetch speakers and line counts
        response = esr.agg_scene_events_by_speaker(show_key, ADMIN_USER, episode_key=episode_key)
        speaker_line_counts = response['scene_events_by_speaker']
        del speaker_line_counts['_ALL_']
        episodes_to_speaker_line_counts[episode_key] = speaker_line_counts
        # transform speakers/line counts to plotly-gantt-friendly span dicts
        for speaker, line_count in speaker_line_counts.items():
            speaker_span = dict(Task=speaker, Start=episode_i, Finish=(episode_i+1), episode_key=episode_key, episode_title=episode_title, 
                                count=line_count, season=episode_season, sequence_in_season=sequence_in_season,
                                info=f'{episode_title} ({line_count} lines)')
            episode_speakers_sequence.append(speaker_span)

        episode_i += 1

    # TODO move this to fig_builder? (where it has to filter rows from the df)
    if limit_cast:
        trimmed_episode_speakers_sequence = []
        for d in episode_speakers_sequence:
            if d['Task'] in show_metadata[show_key.value]['regular_cast'].keys() or d['Task'] in show_metadata[show_key.value]['recurring_cast'].keys():
                trimmed_episode_speakers_sequence.append(d)
        episode_speakers_sequence = trimmed_episode_speakers_sequence

    if overwrite_file:
        file_path = f'{GANTT_DATA_DIR}/{show_key.value}/speaker_gantt_sequence_{show_key.value}.csv'
        print(f'writing speaker gantt sequence dataframe to file_path={file_path}')
        df = pd.DataFrame(episode_speakers_sequence)
        df.to_csv(file_path)

    return episodes_to_speaker_line_counts, episode_speakers_sequence


def generate_series_location_gantt_sequence(show_key: ShowKey, overwrite_file: bool = False) -> tuple[dict, list]:
    '''
    TODO
    '''

    episodes_to_location_counts = {}
    episode_locations_sequence = []

    # limit the superset of locations to those occurring in at least 3 episodes
    response = esr.agg_episodes_by_location(show_key, ADMIN_USER)
    location_episode_counts = response['episodes_by_location']
    del location_episode_counts['_ALL_']
    recurring_locations = [location for location, episode_count in location_episode_counts.items() if episode_count > 2]
    
    # get ordered list of all episodes
    response = esr.fetch_simple_episodes(show_key, ADMIN_USER)
    episodes = response['episodes']

    # for each episode:
    # - fetch all speakers ordered by scene_event count (how many lines they have)
    # - fetch all locations ordered by scene count
    # - transform results of both into lists of span dicts for creating plotly gantt charts
    episode_i = 0
    for episode in episodes:
        episode_key = episode['episode_key']
        episode_title = episode['title']
        episode_season = episode['season']
        sequence_in_season = episode['sequence_in_season']

        # fetch locations and scene counts
        response = esr.agg_scenes_by_location(show_key, episode_key=episode_key)
        location_counts = response['scenes_by_location']
        del location_counts['_ALL_']
        episodes_to_location_counts[episode_key] = location_counts
        # transform locations/counts to plotly-gantt-friendly span dicts
        for location, scene_count in location_counts.items():
            if location in recurring_locations:
                location_span = dict(Task=location, Start=episode_i, Finish=(episode_i+1), episode_key=episode_key, episode_title=episode_title, 
                                     count=scene_count, season=episode_season, sequence_in_season=sequence_in_season,
                                     info=f'{episode_title} ({scene_count} scenes)')
                episode_locations_sequence.append(location_span)

        episode_i += 1

    if overwrite_file:
        file_path = f'{GANTT_DATA_DIR}/{show_key.value}/location_gantt_sequence_{show_key.value}.csv'
        print(f'writing location gantt sequence dataframe to file_path={file_path}')
        df = pd.DataFrame(episode_locations_sequence)
        df.to_csv(file_path)

    return episodes_to_location_counts, episode_locations_sequence


def generate_series_topic_gantt_sequence(show_key: ShowKey, topic_grouping: str, topic_threshold: int, level: str, score_type: str, 
                                         model_vendor: str, model_version: str, overwrite_file: bool) -> tuple[dict, list]:
    '''
    TODO
    '''

    # if not topic_grouping:
    #     topic_grouping = EPISODE_TOPIC_GROUPINGS[0]
    # if not topic_threshold:
    #     topic_threshold = 20
    # if not level:
    #     level = 'leaf'
    # if not score_type:
    #     score_type = 'score'
    # # TODO phasing out dynamic execution of /episode_topic_vector_search in favor of indexed /fetch_episode_topics
    # if not model_vendor:
    #     model_vendor = 'openai'
    # if not model_version:
    #     model_version = '3small'

    episodes_to_topics = {}
    episode_topics_sequence = []
    
    # get ordered list of all episodes
    response = esr.fetch_simple_episodes(show_key, ADMIN_USER)
    episodes = response['episodes']

    # for each episode:
    # - fetch all speakers ordered by scene_event count (how many lines they have)
    # - fetch all locations ordered by scene count
    # - transform results of both into lists of span dicts for creating plotly gantt charts
    episode_i = 0
    for episode in episodes:
        episode_key = episode['episode_key']
        episode_title = episode['title']
        episode_season = episode['season']
        sequence_in_season = episode['sequence_in_season']

        # fetch topics and scores
        response = esr.fetch_episode_topics(show_key, episode_key, topic_grouping, model_vendor, model_version, ADMIN_USER)
        topics = response['episode_topics']
        if len(topics) > topic_threshold:
            topics = topics[:topic_threshold]
        simple_topics = [dict(topic_key=t['topic_key'], score=t[score_type]) for t in topics]
        simple_topics = sorted(simple_topics, key=itemgetter('score'), reverse=True)
        episodes_to_topics[episode_key] = simple_topics
        # transform topics/scores to plotly-gantt-friendly span dicts
        for i in range(len(simple_topics)):
            topic_key = simple_topics[i]['topic_key']
            topic_cat = topic_key.split('.')[0]
            topic_span = dict(Task=topic_key, Start=episode_i, Finish=(episode_i+1), episode_key=episode_key, episode_title=episode_title, 
                              rank=i, topic_cat=topic_cat, season=episode_season, sequence_in_season=sequence_in_season,
                              info=f'{episode_title} (#{i+1} topic)')
            episode_topics_sequence.append(topic_span)

        episode_i += 1

    if overwrite_file:
        file_path = f'{GANTT_DATA_DIR}/{show_key.value}/topic_gantt_sequence_{show_key.value}_{topic_grouping}_{score_type}.csv'
        print(f'writing topic gantt sequence dataframe to file_path={file_path}')
        df = pd.DataFrame(episode_topics_sequence)
        df.to_csv(file_path)

    return episodes_to_topics, episode_topics_sequence
