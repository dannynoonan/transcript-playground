import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime as dt

from app.auth import ADMIN_USER
import app.page_builder_service.page_components as pc
from app.page_callbacks.character_listing_callbacks import *
import app.routers.es_read_router as esr
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

    series_speaker_scene_counts_response = esr.agg_scenes_by_speaker(ShowKey(show_key), ADMIN_USER)
    series_summary['scene_count'] = series_speaker_scene_counts_response['scenes_by_speaker']['_ALL_']

    series_speakers_response = esr.agg_scene_events_by_speaker(ShowKey(show_key), ADMIN_USER)
    series_summary['line_count'] = series_speakers_response['scene_events_by_speaker']['_ALL_']

    series_speaker_word_counts_response = esr.agg_dialog_word_counts(ShowKey(show_key), ADMIN_USER)
    series_summary['word_count'] = int(series_speaker_word_counts_response['dialog_word_counts']['_ALL_'])

    series_speaker_episode_counts_response = esr.agg_episodes_by_speaker(ShowKey(show_key), ADMIN_USER)
    all_series_speakers = list(series_speaker_episode_counts_response['episodes_by_speaker'].keys())
    
    # all seasons
    all_seasons_response = esr.list_seasons(ShowKey(show_key), ADMIN_USER)
    all_seasons = all_seasons_response['seasons']
    series_summary['season_count'] = len(all_seasons)

    # all episodes
    all_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key), ADMIN_USER)
    all_episodes = all_episodes_response['episodes']
    series_summary['episode_count'] = len(all_episodes)

    display_page_end_ts = dt.now()
    display_page_duration = display_page_end_ts - display_page_start_ts
    utils.hilite_in_logs(f'LAYOUT: /character_listing/{show_key} at ts={display_page_end_ts} duration={display_page_duration}')
    ########################## TODO END DATA PRE-AMBLE ##########################


    # generate navbar
    navbar = pc.generate_navbar(show_key, all_seasons)

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
