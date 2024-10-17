import dash_bootstrap_components as dbc
from dash import dash_table
import pandas as pd

import app.data_service.matrix_operations as mxop
import app.es.es_read_router as esr
import app.page_builder_service.page_components as pc
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


def generate_season_episodes_accordion_items(show_key: str, all_season_dicts: dict, speaker_color_map: dict) -> list:
    season_accordion_items = []

    for season, season_dict in all_season_dicts.items():
        # label for collapsed season accordion item
        season_title_text = f"Season {season} ({season_dict['air_date_begin']} — {season_dict['air_date_end']}): {len(season_dict['episodes'])} episodes"

        # episode listing datatable for expanded season accordion item
        season_episodes_dt = generate_season_episodes_dt(show_key, season_dict['episodes'])
        
        # recurring speaker datatable
        recurring_speaker_cols = ['character', 'lines']
        recurring_speaker_df = pd.DataFrame(season_dict['speaker_line_counts'].items(), columns=recurring_speaker_cols)
        speaker_list = list(season_dict['speaker_line_counts'].keys())
        recurring_speaker_dt = pc.pandas_df_to_dash_dt(recurring_speaker_df, recurring_speaker_cols, 'character', speaker_list, speaker_color_map,
                                                       numeric_precision_overrides={'lines': 0})

        # recurring location datatable
        recurring_location_cols = ['location', 'scenes']
        recurring_location_df = pd.DataFrame(season_dict['location_counts'].items(), columns=recurring_location_cols)
        locations_list = list(season_dict['location_counts'].keys())
        bg_color_map = {loc:'DarkSlateBlue' for loc in locations_list}
        recurring_location_dt = pc.pandas_df_to_dash_dt(recurring_location_df, recurring_location_cols, 'location', locations_list, bg_color_map,
                                                        numeric_precision_overrides={'scenes': 0})

        # combine elements into accordion item dash object
        accordion_children = [
            dbc.Row([
                dbc.Col(md=8, children=[season_episodes_dt]),
                dbc.Col(md=2, children=[recurring_speaker_dt]),
                dbc.Col(md=2, children=[recurring_location_dt])
            ])
        ]
        season_accordion_item = dbc.AccordionItem(title=season_title_text, item_id=season, children=accordion_children)
        season_accordion_items.append(season_accordion_item)

    return season_accordion_items


def generate_season_episodes_dt(show_key: str, episodes: list) -> dash_table.DataTable:
    episodes_df = pd.DataFrame(episodes)

    # field naming and processing
    episodes_df['title'] = episodes_df.apply(lambda x: pc.link_to_episode(show_key, x['episode_key'], x['title']), axis=1)
    episodes_df['focal_characters'] = episodes_df['focal_speakers'].apply(lambda x: ', '.join(x))
    episodes_df['genres'] = episodes_df.apply(lambda x: mxop.flatten_df_topics(x['topics_universal_tfidf'], parent_only=True), axis=1)
    episodes_df['air_date'] = episodes_df['air_date'].apply(lambda x: x[:10])
    episodes_df.rename(columns={'sequence_in_season': 'episode'}, inplace=True) 

    # table display input
    display_cols = ['episode', 'title', 'air_date', 'focal_characters', 'genres']
    episode_list = [str(e) for e in list(episodes_df['episode'].unique())]
    bg_color_map = {e:'Maroon' for e in episode_list}

    # convert to dash datatable
    episodes_dt = pc.pandas_df_to_dash_dt(episodes_df, display_cols, 'episode', episode_list, bg_color_map,
                                          numeric_precision_overrides={'episode': 0}, md_cols=['title'])

    return episodes_dt


# TODO this is an exact copy of flatten_and_format_cluster_df from dash.components
def flatten_and_format_cluster_df(show_key: str, clusters_df: pd.DataFrame) -> pd.DataFrame:
    '''
    TODO Holy smackers does this need to be cleaned up. Amazingly it sorta works against two different cluster data sets, but either
    (a) needs to be made more generic or (b) any usage of it must share common column names and data types
    '''

    # reformat columns, sort table
    clusters_df['air_date'] = clusters_df['air_date'].apply(lambda x: x[:10])
    if 'focal_speakers' in clusters_df.columns:
        clusters_df['focal_speakers'] = clusters_df['focal_speakers'].apply(lambda x: ", ".join(x))
    if 'focal_locations' in clusters_df.columns:
        clusters_df['focal_locations'] = clusters_df['focal_locations'].apply(lambda x: ", ".join(x))
    clusters_df['title'] = clusters_df.apply(lambda x: pc.link_to_episode(show_key, x['episode_key'], x['title']), axis=1)
    clusters_df.sort_values(['cluster', 'season', 'sequence_in_season'], inplace=True)

    # rename columns for display
    clusters_df.rename(columns={'sequence_in_season': 'episode', 'scene_count': 'scenes'}, inplace=True)

    clusters_df.drop('episode_key', axis=1, inplace=True) 
    # TODO stop populating this color column, row color is set within dash datatable using style_data_conditional filter_query
    clusters_df.drop('cluster_color', axis=1, inplace=True) 

    return clusters_df
