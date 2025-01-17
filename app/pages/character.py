import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime as dt
from operator import itemgetter

from app.auth import ADMIN_USER
from app.nlp.nlp_metadata import OPENAI_EMOTIONS
import app.page_builder_service.character_page_service as cps
from app.page_callbacks.character_callbacks import *
import app.page_builder_service.page_components as pc
import app.routers.es_read_router as esr
from app.show_metadata import ShowKey, SPEAKER_TOPIC_GROUPINGS
from app import utils


dash.register_page(__name__, path_template='/character/<show_key>/<speaker_key>')


def layout(show_key: str, speaker_key: str) -> html.Div:

    ########################## BEGIN FETCH ON PAGE LOAD ##########################
    display_page_start_ts = dt.now()
    utils.hilite_in_logs(f'PAGE LOAD: /character/{show_key}/{speaker_key} at ts={display_page_start_ts}')

    # all seasons
    all_seasons_response = esr.list_seasons(ShowKey(show_key), ADMIN_USER)
    all_seasons = all_seasons_response['seasons']

    # TODO this is excessive, just hit the speakers index
    # all speakers
    series_speaker_episode_counts_response = esr.agg_episodes_by_speaker(ShowKey(show_key), ADMIN_USER)
    all_series_speakers = list(series_speaker_episode_counts_response['episodes_by_speaker'].keys())
    character_dropdown_options = cps.generate_character_dropdown_options(show_key, all_series_speakers)

    # core speaker data
    speaker_es_response = esr.fetch_speaker(ShowKey(show_key), speaker_key, ADMIN_USER, include_seasons=True, include_episodes=True)
    if 'speaker' not in speaker_es_response:
        raise Exception(f'Failed to find speaker_key={speaker_key}')
    speaker = speaker_es_response['speaker']
    speaker_episodes = []
    if 'episodes' in speaker:
        speaker_episodes = speaker['episodes']
    speaker_seasons = []
    if 'seasons' in speaker:
        speaker_seasons = speaker['seasons']
    actor_names = []
    if 'actor_names' in speaker:
        actor_names = speaker['actor_names']
    alt_names = []
    if 'alt_names' in speaker:
        alt_names = [name for name in speaker['alt_names'] if name.upper() != speaker_key]

    # inject full spectrum of speaker topics
    child_topics_by_grouping = {}
    parent_topics_by_grouping = {}
    for topic_grouping in SPEAKER_TOPIC_GROUPINGS:
        child_topics_response = esr.fetch_speaker_topics(speaker_key, ShowKey(show_key), topic_grouping, ADMIN_USER, level='child')
        child_topics_by_grouping[topic_grouping] = child_topics_response['speaker_topics']
        parent_topics_response = esr.fetch_speaker_topics(speaker_key, ShowKey(show_key), topic_grouping, ADMIN_USER, level='parent')
        parent_topics_by_grouping[topic_grouping] = parent_topics_response['speaker_topics']

    # co-occurring speakers
    co_occ_speakers_by_episode = esr.agg_episodes_by_speaker(ShowKey(show_key), ADMIN_USER, other_speaker=speaker_key)
    co_occ_speakers_by_scene = esr.agg_scenes_by_speaker(ShowKey(show_key), ADMIN_USER, other_speaker=speaker_key)
	# TODO refactor this to generically handle dicts threading together
    other_speakers = {}
    for other_speaker, episode_count in co_occ_speakers_by_episode['episodes_by_speaker'].items():
        if other_speaker not in other_speakers:
            other_speakers[other_speaker] = {}
            other_speakers[other_speaker]['other_speaker'] = other_speaker
        other_speakers[other_speaker]['episode_count'] = episode_count
    for other_speaker, scene_count in co_occ_speakers_by_scene['scenes_by_speaker'].items():
        if other_speaker not in other_speakers:
            other_speakers[other_speaker] = {}
            other_speakers[other_speaker]['other_speaker'] = other_speaker
        other_speakers[other_speaker]['scene_count'] = scene_count
    del(other_speakers[speaker_key])
    del(other_speakers['_ALL_'])
    # TODO shouldn't I be able to sort on a key for a dict within a dict
    other_speaker_dicts = other_speakers.values()
    other_speaker_agg_composite = sorted(other_speaker_dicts, key=itemgetter('episode_count'), reverse=True)

    # similar speakers
    speaker_mlt_response = esr.speaker_mlt_vector_search(ShowKey(show_key), speaker_key, ADMIN_USER)
    speaker_mlt_aggs = speaker_mlt_response['all_speaker_matches']
    # TODO struggling to preserve season and episode sorting
    speaker_mlt_series_matches = speaker_mlt_response['matches_by_speaker_series_embedding']
    speaker_mlt_season_matches = speaker_mlt_response['matches_by_speaker_season_embedding']
    speaker_mlt_episode_matches = speaker_mlt_response['matches_by_speaker_episode_embedding']

    # TODO speaker-specific search

    display_page_end_ts = dt.now()
    display_page_duration = display_page_end_ts - display_page_start_ts
    utils.hilite_in_logs(f'LAYOUT: /character/{show_key}/{speaker_key} at ts={display_page_end_ts} duration={display_page_duration}')
    ##################### END FETCH ON PAGE LOAD #####################

    navbar = pc.generate_navbar(show_key, all_seasons)

    content = html.Div([
        # page storage
        dcc.Store(id='show-key', data=show_key),
        dcc.Store(id='all-seasons', data=all_seasons),
        dcc.Store(id='all-series-speakers', data=all_series_speakers),
        dcc.Store(id='speaker-episodes', data=speaker_episodes),
        dcc.Store(id='speaker-seasons', data=speaker_seasons),
        dcc.Store(id='character-summary', data=speaker_key),
        dcc.Store(id='character-actor-names', data=actor_names),
        dcc.Store(id='character-alt-names', data=alt_names),

        # page display
        navbar,
        dbc.Card(className="bg-dark", children=[

            # character summary / listing dropdown
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=10, children=[
                        html.H3(className="text-white", children=[html.Span(id='character-summary')]),
                        html.H5(className="text-white", style={'display': 'flex'}, children=[
                            html.Div(style={"margin-right": "30px"}, children=[
                                "a.k.a. ", html.B(id='character-alt-names'),
                            ]),
                        ]),
                        html.H5(className="text-white", style={'display': 'flex'}, children=[
                            html.Div(style={"margin-right": "30px"}, children=[
                                "Played by ", html.B(id='character-actor-names'),
                            ]),
                        ]),
                        html.H5(className="text-white", style={'display': 'flex'}, children=[
                            html.Div(style={"margin-right": "30px"}, children=[
                                html.B(id='character-season-count'), " seasons, ", html.B(id='character-episode-count'), " episodes, ", html.B(id='character-scene-count'), " scenes, ", html.B(id='character-line-count'), " lines, ", html.B(id='character-word-count'), " words",
                            ]),
                            html.Div(style={"margin-right": "30px"}, children=[
                                "MBTI: ", html.Span(id='character-top-mbti')
                            ]),
                            html.Div(style={"margin-right": "10px"}, children=[
                                "D&D: ", html.Span(id='character-top-dnda')
                            ]),
                        ]),
                    ]),
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Character: ", dcc.Dropdown(id="speaker-key", options=all_series_speakers, value=speaker_key)
                        ]),
                    ]),
                ]),
            ]),

            # # character episodes
            # dbc.CardBody([
            #     html.H3("Characters in episode"),
            #     dbc.Row([
            #         dbc.Col(md=5, children=[
            #             html.Div(id="speaker-summary-dt"),
            #             html.Br(),
            #             html.Div(dcc.Graph(id="speaker-frequency-bar-chart")),
            #         ]),
            #         dbc.Col(md=7, children=[
            #             html.Div(dcc.Graph(id="speaker-3d-network-graph")),
            #             dcc.RadioItems(
            #                 id="scale-by",
            #                 className="text-white", 
            #                 options=['scenes', 'lines', 'words'],
            #                 value='lines',
            #                 inputStyle={"margin-left": "12px", "margin-right": "4px"},
            #                 style={"display": "flex", "padding-bottom": "0"}
            #             ),
            #         ]),
            #     ]),
            # ]),
        ])
    ])

    return content
