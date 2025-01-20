import pandas as pd

from app.app_metadata import ANIMATION_DATA_DIR
from app.auth import ADMIN_USER
import app.routers.es_read_router as esr
from app.show_metadata import ShowKey, show_metadata


def generate_speaker_line_chart_sequences(show_key: ShowKey, overwrite_file: bool = False) -> list:
    '''
    TODO
    '''

    # TODO distinguish between regular and recurring cast?
    speakers = list(show_metadata[show_key.value]['regular_cast'].keys()) + list(show_metadata[show_key.value]['recurring_cast'].keys())

    speaker_series_agg_word_counts = {spkr:0 for spkr in speakers}
    speaker_series_agg_line_counts = {spkr:0 for spkr in speakers}
    speaker_series_agg_scene_counts = {spkr:0 for spkr in speakers}
    speaker_series_agg_episode_counts = {spkr:0 for spkr in speakers}

    series_agg_word_count = 0
    series_agg_line_count = 0
    series_agg_scene_count = 0
    series_agg_episode_count = 0

    # get ordered list of all episodes
    response = esr.fetch_simple_episodes(show_key, ADMIN_USER)
    episodes = response['episodes']
    
    speaker_episode_rows = []
    episode_i = 0
    curr_season = None
    for episode in episodes:
        episode_key = str(episode['episode_key'])
        episode_title = episode['title']
        season = episode['season']
        sequence_in_season = episode['sequence_in_season']

        if not curr_season or season != curr_season:
            curr_season = season
            season_agg_word_count = 0
            season_agg_line_count = 0
            season_agg_scene_count = 0
            season_agg_episode_count = 0
            speaker_season_agg_word_counts = {spkr:0 for spkr in speakers}
            speaker_season_agg_line_counts = {spkr:0 for spkr in speakers}
            speaker_season_agg_scene_counts = {spkr:0 for spkr in speakers}
            speaker_season_agg_episode_counts = {spkr:0 for spkr in speakers}

        season_agg_episode_count += 1
        series_agg_episode_count += 1

        # fetch speakers and word counts
        word_count_agg_response = esr.agg_dialog_word_counts(show_key, ADMIN_USER, episode_key=episode_key)
        speaker_word_counts = word_count_agg_response['dialog_word_counts']
        episode_word_count = speaker_word_counts['_ALL_']
        season_agg_word_count += episode_word_count
        series_agg_word_count += episode_word_count
        # fetch speakers and line counts
        scene_event_agg_response = esr.agg_scene_events_by_speaker(show_key, ADMIN_USER, episode_key=episode_key)
        speaker_line_counts = scene_event_agg_response['scene_events_by_speaker']
        episode_line_count = speaker_line_counts['_ALL_']
        season_agg_line_count += episode_line_count
        series_agg_line_count += episode_line_count
        # fetch speakers and scene/episode counts
        scene_agg_response = esr.agg_scenes_by_speaker(show_key, ADMIN_USER, episode_key=episode_key)
        speaker_scene_counts = scene_agg_response['scenes_by_speaker']
        episode_scene_count = speaker_scene_counts['_ALL_']
        season_agg_scene_count += episode_scene_count
        series_agg_scene_count += episode_scene_count
        # episodes_to_speaker_counts[episode_key] = speaker_scene_counts.keys()

        for speaker in speakers:
            if speaker in speaker_word_counts:
                # speaker_episode_row = {}
                word_count = speaker_word_counts[speaker] 
                line_count = speaker_line_counts[speaker]
                scene_count = speaker_scene_counts[speaker]
                # increment agg speaker counts
                speaker_season_agg_word_counts[speaker] += word_count
                speaker_series_agg_word_counts[speaker] += word_count
                speaker_season_agg_line_counts[speaker] += line_count
                speaker_series_agg_line_counts[speaker] += line_count
                speaker_season_agg_scene_counts[speaker] += scene_count
                speaker_series_agg_scene_counts[speaker] += scene_count
                speaker_season_agg_episode_counts[speaker] += 1
                speaker_series_agg_episode_counts[speaker] += 1
            else:
                word_count = 0 
                line_count = 0
                scene_count = 0

            # init speaker_episode_row
            speaker_episode_row = dict(
                speaker=speaker,
                episode_key=episode_key,
                episode_i=episode_i, 
                episode_title=episode_title,
                season=season,
                sequence_in_season=sequence_in_season,
                word_count=word_count, 
                line_count=line_count, 
                scene_count=scene_count)
            # speaker X counts as a % of episode X count
            speaker_episode_row['word_count_pct_of_episode'] = word_count / episode_word_count
            speaker_episode_row['line_count_pct_of_episode'] = line_count / episode_line_count
            speaker_episode_row['scene_count_pct_of_episode'] = scene_count / episode_scene_count
            # season agg speaker X counts as a % of season agg X count
            speaker_episode_row['word_count_pct_of_season'] = speaker_season_agg_word_counts[speaker] / season_agg_word_count
            speaker_episode_row['line_count_pct_of_season'] = speaker_season_agg_line_counts[speaker] / season_agg_line_count
            speaker_episode_row['scene_count_pct_of_season'] = speaker_season_agg_scene_counts[speaker] / season_agg_scene_count
            speaker_episode_row['episode_count_pct_of_season'] = speaker_season_agg_episode_counts[speaker] / season_agg_episode_count
            # overall agg speaker X counts as a % of overall agg X count
            speaker_episode_row['word_count_pct_of_series'] = speaker_series_agg_word_counts[speaker] / series_agg_word_count
            speaker_episode_row['line_count_pct_of_series'] = speaker_series_agg_line_counts[speaker] / series_agg_line_count
            speaker_episode_row['scene_count_pct_of_series'] = speaker_series_agg_scene_counts[speaker] / series_agg_scene_count
            speaker_episode_row['episode_count_pct_of_series'] = speaker_series_agg_episode_counts[speaker] / series_agg_episode_count
            
            speaker_episode_row['info'] = f'{speaker} in {episode_title}: {scene_count} scenes, {line_count} lines, {word_count} words'
            speaker_episode_rows.append(speaker_episode_row)

        episode_i += 1

    if overwrite_file:
        file_path = f'{ANIMATION_DATA_DIR}/{show_key.value}/speaker_episode_aggs_{show_key.value}.csv'
        print(f'writing speaker word/line/scene/episode counts and aggs dataframe to file_path={file_path}')
        df = pd.DataFrame(speaker_episode_rows)
        df.to_csv(file_path)

    return speaker_episode_rows


def generate_location_line_chart_sequences(show_key: ShowKey, overwrite_file: bool = False) -> list:
    '''
    TODO
    '''
    
    response = esr.agg_scenes_by_location(show_key, ADMIN_USER)
    locations = response['scenes_by_location']
    top_locations = [location for location, count in locations.items() if count > 10]
    location_series_agg_scene_counts = {location:0 for location in top_locations}
    location_series_agg_episode_counts = {location:0 for location in top_locations}

    series_agg_scene_count = 0
    series_agg_episode_count = 0

    # get ordered list of all episodes
    response = esr.fetch_simple_episodes(show_key, ADMIN_USER)
    episodes = response['episodes']
    
    location_episode_rows = []
    episode_i = 0
    curr_season = None
    for episode in episodes:
        episode_key = episode['episode_key']
        episode_title = episode['title']
        season = episode['season']
        sequence_in_season = episode['sequence_in_season']

        if not curr_season or season != curr_season:
            curr_season = season
            season_agg_scene_count = 0
            season_agg_episode_count = 0
            location_season_agg_scene_counts = {location:0 for location in top_locations}
            location_season_agg_episode_counts = {location:0 for location in top_locations}

        season_agg_episode_count += 1
        series_agg_episode_count += 1

        # fetch locations and scene/episode counts
        scene_agg_response = esr.agg_scenes_by_location(show_key, ADMIN_USER, episode_key=episode_key)
        location_scene_counts = scene_agg_response['scenes_by_location']
        episode_scene_count = location_scene_counts['_ALL_']
        del location_scene_counts['_ALL_']
        season_agg_scene_count += episode_scene_count
        series_agg_scene_count += episode_scene_count
        # episodes_to_speaker_counts[episode_key] = speaker_scene_counts.keys()

        for location in top_locations:
            if location in location_scene_counts:
                # location_episode_row = {}

                scene_count = location_scene_counts[location]
                # increment agg location counts
                location_season_agg_scene_counts[location] += scene_count
                location_series_agg_scene_counts[location] += scene_count
                location_season_agg_episode_counts[location] += 1
                location_series_agg_episode_counts[location] += 1
            else:
                scene_count = 0

            # init location_episode_row
            location_episode_row = dict(
                location=location,
                episode_i=episode_i, 
                episode_title=episode_title,
                season=season,
                sequence_in_season=sequence_in_season,
                scene_count=scene_count)
            # location scene counts as a % of episode scene count
            location_episode_row['scene_count_pct_of_episode'] = scene_count / episode_scene_count
            # season agg speaker scene/episode counts as a % of season agg scene/episode count
            location_episode_row['scene_count_pct_of_season'] = location_season_agg_scene_counts[location] / season_agg_scene_count
            location_episode_row['episode_count_pct_of_season'] = location_season_agg_episode_counts[location] / season_agg_episode_count
            # overall agg speaker scene/episode counts as a % of overall agg scene/episode count
            location_episode_row['scene_count_pct_of_series'] = location_series_agg_scene_counts[location] / series_agg_scene_count
            location_episode_row['episode_count_pct_of_series'] = location_series_agg_episode_counts[location] / series_agg_episode_count
            
            location_episode_row['info'] = f'{location} in {episode_title}: {scene_count} scenes'
            location_episode_rows.append(location_episode_row)

        episode_i += 1

    if overwrite_file:
        file_path = f'{ANIMATION_DATA_DIR}/{show_key.value}/location_episode_aggs_{show_key.value}.csv'
        print(f'writing location scene/episode counts and aggs dataframe to file_path={file_path}')
        df = pd.DataFrame(location_episode_rows)
        df.to_csv(file_path)

    return location_episode_rows
