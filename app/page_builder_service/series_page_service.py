import app.es.es_read_router as esr
from app.show_metadata import ShowKey
from app import utils



def generate_series_summary(show_key: str) -> tuple[dict, dict]:
    series_summary = {}
    series_summary['series_title'] = 'Star Trek: The Next Generation'

    series_speaker_scene_counts_response = esr.agg_scenes_by_speaker(ShowKey(show_key))
    series_summary['scene_count'] = series_speaker_scene_counts_response['scenes_by_speaker']['_ALL_']

    series_speakers_response = esr.agg_scene_events_by_speaker(ShowKey(show_key))
    series_summary['line_count'] = series_speakers_response['scene_events_by_speaker']['_ALL_']

    series_speaker_word_counts_response = esr.agg_dialog_word_counts(ShowKey(show_key))
    series_summary['word_count'] = int(series_speaker_word_counts_response['dialog_word_counts']['_ALL_'])

    # series_locations_response = esr.agg_scenes_by_location(ShowKey(show_key))
    # location_count = series_locations_response['location_count']

    episodes_by_season_response = esr.list_simple_episodes_by_season(ShowKey(show_key))
    episodes_by_season = episodes_by_season_response['episodes_by_season']

    series_summary['season_count'] = len(episodes_by_season)
    series_summary['episode_count'] = 0

    return series_summary, episodes_by_season


def generate_all_season_episode_data(show_key: str, episodes_by_season: dict, series_summary: dict):
    all_season_episode_data = {}
    first_episode_in_series = None
    last_episode_in_series = None

    for season, episodes in episodes_by_season.items():
        season_episode_data_dict = {}

        season_episode_data_dict['episodes'] = episodes
        season_episode_count = len(episodes_by_season[season])
        series_summary['episode_count'] += len(episodes_by_season[season])
        
        scenes_by_location_response = esr.agg_scenes_by_location(ShowKey(show_key), season=season)
        season_episode_data_dict['location_count'] = scenes_by_location_response['location_count']
        season_episode_data_dict['location_counts'] = utils.truncate_dict(scenes_by_location_response['scenes_by_location'], season_episode_count, start_index=1)

        scene_events_by_speaker_response = esr.agg_scene_events_by_speaker(ShowKey(show_key), season=season)
        season_episode_data_dict['line_count'] = scene_events_by_speaker_response['scene_events_by_speaker']['_ALL_']
        season_episode_data_dict['speaker_line_counts'] = utils.truncate_dict(scene_events_by_speaker_response['scene_events_by_speaker'], season_episode_count, start_index=1)
        
        scenes_by_speaker_response = esr.agg_scenes_by_speaker(ShowKey(show_key), season=season)
        season_episode_data_dict['scene_count'] = scenes_by_speaker_response['scenes_by_speaker']['_ALL_']

        episodes_by_speaker_response = esr.agg_episodes_by_speaker(ShowKey(show_key), season=season)
        season_episode_data_dict['speaker_count'] = episodes_by_speaker_response['speaker_count']

        word_counts_response = esr.agg_dialog_word_counts(ShowKey(show_key), season=season)
        season_episode_data_dict['word_count'] = int(word_counts_response['dialog_word_counts']['_ALL_'])

        # air_date range
        first_episode_in_season = episodes_by_season[season][0]
        last_episode_in_season = episodes_by_season[season][-1]
        season_episode_data_dict['air_date_begin'] = first_episode_in_season['air_date'][:10]
        season_episode_data_dict['air_date_end'] = last_episode_in_season['air_date'][:10]
        if not first_episode_in_series:
            first_episode_in_series = episodes_by_season[season][0]
        last_episode_in_series = episodes_by_season[season][-1]
        all_season_episode_data[season] = season_episode_data_dict

    series_summary['air_date_begin'] = first_episode_in_series['air_date'][:10]
    series_summary['air_date_end'] = last_episode_in_series['air_date'][:10]

    return all_season_episode_data


def get_parent_topics_for_grouping(topic_grouping: str):
    parent_topics = []
    topic_grouping_response = esr.fetch_topic_grouping(topic_grouping)
    for t in topic_grouping_response['topics']:
        # only process topics that have parents (ignore the parents themselves)
        if not t['parent_key']:
            parent_topics.append(t['topic_key'])
    
    return parent_topics
