import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime as dt

import app.es.es_read_router as esr
import app.fig_meta.color_meta as cm
import app.page_builder_service.series_page_service as sps
from app.page_callbacks.series_callbacks import *
import app.page_builder_service.page_components as pc
from app.show_metadata import ShowKey
from app import utils


dash.register_page(__name__, path_template='/series/<show_key>')


def layout(show_key: str) -> html.Div:

    display_page_start_ts = dt.now()
    utils.hilite_in_logs(f'PAGE LOAD: /series/{show_key} at ts={display_page_start_ts}')

    ##################### BEGIN FETCH ON PAGE LOAD #####################
    
    # series speaker data and color map
    series_speaker_episode_counts_response = esr.agg_episodes_by_speaker(ShowKey(show_key))
    all_series_speakers = list(series_speaker_episode_counts_response['episodes_by_speaker'].keys())
    speaker_color_map = cm.generate_speaker_color_discrete_map(show_key, all_series_speakers)
    
    # series summary and season episode listing data
    series_summary, episodes_by_season = sps.generate_series_summary(show_key)
    all_season_episode_data = sps.generate_all_season_episode_data(show_key, episodes_by_season, series_summary)
    season_accordion_items = sps.generate_season_episodes_accordion_items(show_key, all_season_episode_data, speaker_color_map)

    # episode data
    simple_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
    all_simple_episodes = simple_episodes_response['episodes']

    # topic listing data
    universal_genres_parent_topics = sps.get_parent_topics_for_grouping('universalGenres')

    ##################### END FETCH ON PAGE LOAD ##################### 

    display_page_end_ts = dt.now()
    display_page_duration = display_page_end_ts - display_page_start_ts
    utils.hilite_in_logs(f'LAYOUT: /episode/{show_key} at ts={display_page_end_ts} duration={display_page_duration}')

    # generate navbar
    all_seasons = list(episodes_by_season.keys())
    navbar = pc.generate_navbar(show_key, all_seasons)

    # other page assets
    wordcloud_img = f'/static/wordclouds/{show_key}/{show_key}_SERIES.png'

    # define content div
    content = html.Div([
        # page storage
        dcc.Store(id='show-key', data=show_key),
        dcc.Store(id='all-simple-episodes', data=all_simple_episodes),
        dcc.Store(id='all-seasons', data=all_seasons),
        dcc.Store(id='speaker-color-map', data=speaker_color_map),

        # page display
        navbar,
        dbc.Card(className="bg-dark", children=[

            # series summary
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=12, children=[
                        html.H3(className="text-white", children=[
                            html.B(series_summary['series_title']), " (", series_summary['air_date_begin'], " — ", series_summary['air_date_end'], ")"
                        ]),
                        html.H5(className="text-white", style={'display': 'flex'}, children=[
                            html.Div(style={"margin-right": "30px"}, children=[
                                html.B(series_summary['season_count']), " seasons, ", html.B(series_summary['episode_count']), " episodes, ", 
                                html.B(series_summary['scene_count']), " scenes, ", html.B(series_summary['line_count']), " lines, ", html.B(series_summary['word_count']), " words",
                            ]),
                        ]),
                    ]),
                    # dbc.Col(md=2),
                    # dbc.Col(md=2, children=[
                    #     html.Div([
                    #         "Show: ", dcc.Dropdown(id="show-key", options=[show_key], value=show_key)
                    #     ]),
                    # ]),
                ]),
            ]),

            # series continuity timelines for character / location / topic / search
            dbc.CardBody([
                html.H3("Continuity over course of series"),
                dbc.Row([
                    dbc.Col(md=12, children=[
                        dbc.Tabs(className="nav nav-tabs", children=[
                            dbc.Tab(label="Characters", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="series-speakers-gantt"),
                                ])
                            ]),
                            dbc.Tab(label="Locations", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="series-locations-gantt"),
                                ]),
                            ]),
                            dbc.Tab(label="Genres", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dcc.RadioItems(
                                    id="series-topics-gantt-score-type",
                                    className="text-white", 
                                    options=[
                                        {'label': 'absolute scoring', 'value': 'score'},
                                        {'label': 'frequency-based scoring', 'value': 'tfidf_score'},
                                    ],
                                    value='score',
                                    inputStyle={"margin-left": "12px", "margin-right": "4px"},
                                    style={"display": "flex", "padding-bottom": "0"}
                                ),
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="series-topics-gantt"),
                                ]),
                            ]),
                            dbc.Tab(label="Dialog search", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=4, children=[
                                        html.Div([
                                            "Search in dialog: ", dbc.Input(id="series-search-qt", type="text", autoFocus=True, debounce=True),
                                        ]),
                                    ]),
                                ]),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col(md=12, children=[
                                        html.H5(children=["Results: ", html.Span(id='series-search-response-text')]),
                                        dcc.Graph(id="series-search-results-gantt-new"),
                                        html.Br(),
                                        html.Div(id="series-search-results-dt"),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),

            # series episode scatter grid 
            dbc.CardBody([
                html.H3("Season-episode grid — Prominent characters, locations, and genres"),
                dbc.Row([
                    dbc.Col(md=8, children=[
                        dcc.RadioItems(
                            id="scatter-grid-hilite",
                            className="text-white", 
                            options=[
                                {'label': 'focal character', 'value': 'focal_speakers'},
                                {'label': 'focal location', 'value': 'focal_locations'},
                                {'label': 'genre (absolute scoring)', 'value': 'topics_universal'},
                                {'label': 'genre (weighted by frequency)', 'value': 'topics_universal_tfidf'},
                            ],
                            value='focal_speakers',
                            inputStyle={"margin-left": "12px", "margin-right": "4px"},
                            style={"display": "flex", "padding-bottom": "0"}
                        ),
                        html.Div(dcc.Graph(id="series-episodes-scatter-grid")),
                    ]),
                    dbc.Col(md=4, children=[
                        html.Img(src=wordcloud_img, width="100%")
                    ]),
                ]),
            ]),

            # series episode listing accordion
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=12, children=[
                        dbc.Accordion(id="accordion", active_item="acc_textarea", children=season_accordion_items),
                        html.Div(id="accordion-contents", className="mt-3"),
                    ]),
                ]),
            ]),

            # series-topic mappings & episode clusters
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=12, children=[
                        dbc.Tabs(className="nav nav-tabs", children=[
                            dbc.Tab(label="Episode genres", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            "Topic grouping: ", 
                                            dcc.Dropdown(
                                                id="series-topic-pie-topic-grouping", 
                                                options=['universalGenres', 'universalGenresGpt35_v2'], 
                                                value='universalGenres')
                                        ]),
                                    ]),
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            "Score type: ", 
                                            dcc.Dropdown(
                                                id="series-topic-pie-score-type", 
                                                options=['score', 'tfidf_score'], 
                                                value='tfidf_score')
                                        ]),
                                    ]),
                                ]),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col(md=6, children=[
                                        html.Div(dcc.Graph(id="series-parent-topic-pie")),
                                    ]),
                                    dbc.Col(md=6, children=[
                                        html.Div(dcc.Graph(id="series-topic-pie")),
                                    ]),
                                ]),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            "List episodes for topic: ", dcc.Dropdown(id="show-series-topic-episodes-dt-for-topic", options=universal_genres_parent_topics)
                                        ]),
                                    ]),
                                ]),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col(md=12, children=[
                                        html.Div(id="series-topic-episodes-dt"),
                                    ]),
                                ]),
                            ]),
                            dbc.Tab(label="Episode clusters", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            "Number of clusters: ", dcc.Dropdown(id="num-clusters", options=[2, 3, 4, 5, 6, 7, 8, 9, 10], value=5)
                                        ]),
                                    ]),
                                ]),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col(md=12, children=[
                                        html.Div(dcc.Graph(id="series-episodes-cluster-scatter")),
                                    ]),
                                ]),
                                dcc.Checklist(
                                    id="show-series-episodes-cluster-dt",
                                    className="text-white", 
                                    options=[
                                        {'label': 'Display as table listing', 'value': 'yes'}
                                    ],
                                    value=[],
                                    inputStyle={"margin-left": "12px", "margin-right": "4px"},
                                    style={"display": "flex", "padding-bottom": "0"}
                                ),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col(md=12, children=[
                                        html.Div(id="series-episodes-cluster-dt"),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),

            # series speaker listing
            dbc.CardBody([
                html.H3("Regular and recurring characters in series"),
                dbc.Row([
                    dbc.Col(md=12, children=[
                        html.Div(id="series-speaker-listing-dt"),
                    ]),
                ]),
            ]),

            # series speaker counts
            dbc.CardBody([
                dbc.Row([
                    html.H3("Character chatter"),
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Tally by: ",
                            dcc.Dropdown(
                                id="speaker-chatter-tally-by",
                                options=['episode', 'scene', 'line', 'word'],
                                value='line',
                            )
                        ]),
                    ]),
                ]),
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dbc.Col(md=6, children=[
                        html.Div([
                            "Season ",
                            html.Br(),
                            dcc.Slider(
                                id="speaker-chatter-season",
                                min=all_seasons[0],
                                max=all_seasons[len(all_seasons)-1],
                                step=None,
                                marks={
                                    int(y): {'label': str(y), 'style': {'transform': 'rotate(45deg)', 'color': 'white'}}
                                    for y in all_seasons
                                },
                                value=all_seasons[0],
                            ),
                            html.Br(),
                            dcc.Graph(id="speaker-season-frequency-bar-chart"),
                        ]),
                    ]),
                    dbc.Col(md=6, children=[
                        html.Div([
                            "Episode ",
                            html.Br(),
                            dcc.Slider(
                                id="speaker-chatter-sequence-in-season",
                                min=1,
                                max=25,
                                step=None,
                                marks={
                                    int(y): {'label': str(y), 'style': {'transform': 'rotate(45deg)', 'color': 'white'}}
                                    for y in range(1,26)
                                },
                                value=1,
                            ),
                            html.Br(),
                            dcc.Graph(id="speaker-episode-frequency-bar-chart"),
                        ]),
                    ]),
                ]),
                html.Br(),
            ]),

            # series-speaker-topic mappings
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=12, children=[
                        dbc.Tabs(className="nav nav-tabs", children=[
                            dbc.Tab(label="Character MBTI temperaments", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=7, children=[
                                        html.Div(dcc.Graph(id="series-speaker-mbti-scatter")),
                                        html.Br(),
                                        dbc.Row([
                                            dbc.Col(md=5, style={"text-align": "right", "color": "white"}, children=['Alt temperaments:']),
                                            dbc.Col(md=4, children=[
                                                dcc.Slider(id="series-mbti-count", min=1, max=4, step=1, value=3),
                                            ]),
                                        ]),
                                    ]),
                                    dbc.Col(md=5, children=[
                                        html.Div(id="series-speaker-mbti-dt"),
                                    ]),
                                ]),
                            ]),
                            dbc.Tab(label="Character D&D alignments", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=7, children=[
                                        html.Div(dcc.Graph(id="series-speaker-dnda-scatter")),
                                        html.Br(),
                                        dbc.Row([
                                            dbc.Col(md=5, style={"text-align": "right", "color": "white"}, children=['Alt alignments:']),
                                            dbc.Col(md=4, children=[
                                                dcc.Slider( id="series-dnda-count", min=1, max=3, step=1, value=2),
                                            ]),
                                        ]),
                                    ]),
                                    dbc.Col(md=5, children=[
                                        html.Div(id="series-speaker-dnda-dt"),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ])
    ])

    return content
