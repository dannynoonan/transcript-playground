from dash import callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

import app.es.es_read_router as esr
import app.data_service.field_flattener as fflat 
import app.fig_meta.color_meta as cm
import app.pages.components as cmp
from app.show_metadata import ShowKey
from app import utils



############ series speaker listing callback
@callback(
    # Output('speaker-qt-display', 'children'),
    Output('speaker-listing-dt', 'children'),
    # Output('speaker-matches-dt', 'children'),
    Input('show-key', 'value'))
    # Input('speaker-qt', 'value')) 
def render_speaker_listing_dt(show_key: str):
    utils.hilite_in_logs(f'callback invoked: render_speaker_listing_dt, show_key={show_key}')   
# def render_speaker_listing_dt(show_key: str, speaker_qt: str):
#     print(f'in render_speaker_listing_dt, show_key={show_key} speaker_qt={speaker_qt}')

    indexed_speakers_response = esr.fetch_indexed_speakers(ShowKey(show_key), extra_fields='topics_mbti')
    indexed_speakers = indexed_speakers_response['speakers']
    indexed_speakers = fflat.flatten_speaker_topics(indexed_speakers, 'mbti', limit_per_speaker=3) 
    indexed_speakers = fflat.flatten_and_refine_alt_names(indexed_speakers, limit_per_speaker=1) 
    
    speakers_df = pd.DataFrame(indexed_speakers)

	# # TODO well THIS is inefficient...
    # indexed_speaker_keys = [s['speaker'] for s in indexed_speakers]
    # speaker_aggs_response = esr.composite_speaker_aggs(show_key)
    # speaker_aggs = speaker_aggs_response['speaker_agg_composite']
    # non_indexed_speakers = [s for s in speaker_aggs if s['speaker'] not in indexed_speaker_keys]

    speaker_names = [s['speaker'] for s in indexed_speakers]

    speakers_df.rename(columns={'speaker': 'character', 'scene_count': 'scenes', 'line_count': 'lines', 'word_count': 'words', 'season_count': 'seasons', 
                                'episode_count': 'episodes', 'actor_names': 'actor(s)', 'topics_mbti': 'mbti'}, inplace=True)
    display_cols = ['character', 'aka', 'actor(s)', 'seasons', 'episodes', 'scenes', 'lines', 'words', 'mbti']

    # replace actor nan values with empty string, flatten list into string
    speakers_df['actor(s)'].fillna('', inplace=True)
    speakers_df['actor(s)'] = speakers_df['actor(s)'].apply(lambda x: ', '.join(x))

    speaker_colors = cm.generate_speaker_color_discrete_map(show_key, speaker_names)

    speaker_listing_dt = cmp.pandas_df_to_dash_dt(speakers_df, display_cols, 'character', speaker_names, speaker_colors,
                                                  numeric_precision_overrides={'seasons': 0, 'episodes': 0, 'scenes': 0, 'lines': 0, 'words': 0})

    # print('speaker_listing_dt:')
    # utils.hilite_in_logs(speaker_listing_dt)

    # speaker_matches = []
    # if speaker_qt:
    # 	# speaker_qt_display = speaker_qt
    #     speaker_search_response = esr.search_speakers(speaker_qt, show_key=show_key)
    #     speaker_matches = speaker_search_response['speaker_matches']
    #     speaker_matches_dt = None

    # return speaker_qt, speaker_listing_dt, speaker_matches_dt
    return speaker_listing_dt
