import dash
from dash import callback, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime as dt
import pandas as pd

import app.es.es_read_router as esr
import app.fig_builder.fig_helper as fh 
import app.pages.components as cmp
from app.show_metadata import ShowKey
from app import utils


dash.register_page(__name__, path_template='/character_listing/<show_key>')


def layout(show_key: str) -> html.Div:

    display_page_start_ts = dt.now()
    utils.hilite_in_logs(f'PAGE LOAD: /character_listing/{show_key} at ts={display_page_start_ts}')
    
    ########################## TODO BEGIN DATA PRE-AMBLE ##########################
    # all seasons
    series_summary = {}
    series_summary['series_title'] = 'Star Trek: The Next Generation'

    series_speaker_scene_counts_response = esr.agg_scenes_by_speaker(ShowKey(show_key))
    series_summary['scene_count'] = series_speaker_scene_counts_response['scenes_by_speaker']['_ALL_']

    series_speakers_response = esr.agg_scene_events_by_speaker(ShowKey(show_key))
    series_summary['line_count'] = series_speakers_response['scene_events_by_speaker']['_ALL_']

    series_speaker_word_counts_response = esr.agg_dialog_word_counts(ShowKey(show_key))
    series_summary['word_count'] = int(series_speaker_word_counts_response['dialog_word_counts']['_ALL_'])

    series_speaker_episode_counts_response = esr.agg_episodes_by_speaker(ShowKey(show_key))
    all_series_speakers = list(series_speaker_episode_counts_response['episodes_by_speaker'].keys())
    
    # all seasons
    all_seasons_response = esr.list_seasons(ShowKey(show_key))
    all_seasons = all_seasons_response['seasons']
    series_summary['season_count'] = len(all_seasons)

    # all episodes
    all_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
    all_episodes = all_episodes_response['episodes']
    series_summary['episode_count'] = len(all_episodes)

    display_page_end_ts = dt.now()
    display_page_duration = display_page_end_ts - display_page_start_ts
    utils.hilite_in_logs(f'LAYOUT: /character_listing/{show_key} at ts={display_page_end_ts} duration={display_page_duration}')
    ########################## TODO END DATA PRE-AMBLE ##########################


    # generate navbar
    navbar = cmp.generate_navbar(show_key, all_seasons)

    # define content div
    content = html.Div([
        navbar,
        dbc.Card(className="bg-dark", children=[   

            # series summary
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=8, children=[
                        html.H3(className="text-white", children=[
                            html.B(series_summary['series_title'])
                        ]),
                        html.H5(className="text-white", style={'display': 'flex'}, children=[
                            html.Div(style={"margin-right": "30px"}, children=[
                                html.B(series_summary['season_count']), " seasons, ", html.B(series_summary['episode_count']), " episodes, ", 
                                html.B(series_summary['scene_count']), " scenes, ", html.B(series_summary['line_count']), " lines, ", html.B(series_summary['word_count']), " words",
                            ]),
                        ]),
                    ]),
                    dbc.Col(md=4, children=[
                        dbc.Row([ 
                            dbc.Col(md=6),
                            dbc.Col(md=6, children=[
                                html.Div([
                                    "Show: ", dcc.Dropdown(id="show-key", options=[show_key], value=show_key)
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]), 

            # series speaker listing
            dbc.CardBody([
                html.H3("Characters in series"),
                dbc.Row([
                    dbc.Col(md=12, children=[
                        html.Div(id="speaker-listing-dt"),
                    ]),
                ]),
            ]),

        ])
    ])

    return content









######################################## CHARACTER LISTING CALLBACKS ##########################################

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
    indexed_speakers = fh.flatten_speaker_topics(indexed_speakers, 'mbti', limit_per_speaker=3) 
    indexed_speakers = fh.flatten_and_refine_alt_names(indexed_speakers, limit_per_speaker=1) 
    
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

    speaker_colors = fh.generate_speaker_color_discrete_map(show_key, speaker_names)

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
