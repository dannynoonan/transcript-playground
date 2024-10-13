import dash
from dash import callback, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime as dt
import os
import pandas as pd

import app.es.es_read_router as esr
import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.fig_builder.fig_helper as fh
import app.fig_builder.fig_metadata as fm
import app.fig_builder.plotly_bar as pbar
import app.fig_builder.plotly_gantt as pgantt
import app.fig_builder.plotly_pie as ppie
import app.fig_builder.plotly_scatter as pscat
import app.figdata_manager.color_meta as cm
import app.figdata_manager.gantt_helper as gh
import app.data_service.field_flattener as fflat
import app.nlp.embeddings_factory as ef
from app.nlp.nlp_metadata import OPENAI_EMOTIONS
import app.pages.components as cmp
from app.show_metadata import show_metadata, ShowKey
from app import utils


dash.register_page(__name__, path_template='/series/<show_key>')


def layout(show_key: str) -> html.Div:

    display_page_start_ts = dt.now()
    utils.hilite_in_logs(f'PAGE LOAD: /series/{show_key} at ts={display_page_start_ts}')
    

    ########################## TODO BEGIN DATA PRE-AMBLE ##########################
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

    # series_locations_response = esr.agg_scenes_by_location(ShowKey(show_key))
    # location_count = series_locations_response['location_count']

    episodes_by_season_response = esr.list_simple_episodes_by_season(ShowKey(show_key))
    episodes_by_season = episodes_by_season_response['episodes_by_season']

    all_seasons = list(episodes_by_season.keys())
    series_summary['season_count'] = len(all_seasons)

    series_summary['episode_count'] = 0
    first_episode_in_series = None
    last_episode_in_series = None
    all_season_dicts = {}

    for season, episodes in episodes_by_season.items():
        season_dict = {}

        season_dict['episodes'] = episodes
        season_episode_count = len(episodes_by_season[season])
        series_summary['episode_count'] += len(episodes_by_season[season])
        
        scenes_by_location_response = esr.agg_scenes_by_location(ShowKey(show_key), season=season)
        season_dict['location_count'] = scenes_by_location_response['location_count']
        season_dict['location_counts'] = utils.truncate_dict(scenes_by_location_response['scenes_by_location'], season_episode_count, start_index=1)

        scene_events_by_speaker_response = esr.agg_scene_events_by_speaker(ShowKey(show_key), season=season)
        season_dict['line_count'] = scene_events_by_speaker_response['scene_events_by_speaker']['_ALL_']
        season_dict['speaker_line_counts'] = utils.truncate_dict(scene_events_by_speaker_response['scene_events_by_speaker'], season_episode_count, start_index=1)
        
        scenes_by_speaker_response = esr.agg_scenes_by_speaker(ShowKey(show_key), season=season)
        season_dict['scene_count'] = scenes_by_speaker_response['scenes_by_speaker']['_ALL_']

        episodes_by_speaker_response = esr.agg_episodes_by_speaker(ShowKey(show_key), season=season)
        season_dict['speaker_count'] = episodes_by_speaker_response['speaker_count']

        word_counts_response = esr.agg_dialog_word_counts(ShowKey(show_key), season=season)
        season_dict['word_count'] = int(word_counts_response['dialog_word_counts']['_ALL_'])

        # air_date range
        first_episode_in_season = episodes_by_season[season][0]
        last_episode_in_season = episodes_by_season[season][-1]
        season_dict['air_date_begin'] = first_episode_in_season['air_date'][:10]
        season_dict['air_date_end'] = last_episode_in_season['air_date'][:10]
        if not first_episode_in_series:
            first_episode_in_series = episodes_by_season[season][0]
        last_episode_in_series = episodes_by_season[season][-1]
        all_season_dicts[season] = season_dict

    series_summary['air_date_begin'] = first_episode_in_series['air_date'][:10]
    series_summary['air_date_end'] = last_episode_in_series['air_date'][:10]

    universal_genres_parent_topics = []
    topic_grouping_response = esr.fetch_topic_grouping('universalGenres')
    for t in topic_grouping_response['topics']:
        # only process topics that have parents (ignore the parents themselves)
        if not t['parent_key']:
            universal_genres_parent_topics.append(t['topic_key'])

    # series_speaker_names = list(show_metadata[show_key]['regular_cast'].keys()) + list(show_metadata[show_key]['recurring_cast'].keys())
    speaker_color_map = cm.generate_speaker_color_discrete_map(show_key, all_series_speakers)

    display_page_end_ts = dt.now()
    display_page_duration = display_page_end_ts - display_page_start_ts
    utils.hilite_in_logs(f'LAYOUT: /episode/{show_key} at ts={display_page_end_ts} duration={display_page_duration}')
    ########################## TODO END DATA PRE-AMBLE ##########################




    # generate navbar
    navbar = cmp.generate_navbar(show_key, all_seasons)
    wordcloud_img = f'/static/wordclouds/{show_key}/{show_key}_SERIES.png'

    # generate season episode listing datatables in accordion items
    season_accordion_items = cmp.generate_season_episodes_accordion_items(all_season_dicts, speaker_color_map)

    # define content div
    content = html.Div([
        navbar,
        dbc.Card(className="bg-dark", children=[

            # series summary
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=8, children=[
                        html.H3(className="text-white", children=[
                            html.B(series_summary['series_title']), " (", series_summary['air_date_begin'], " — ", series_summary['air_date_end'], ")"
                        ]),
                        html.H5(className="text-white", style={'display': 'flex'}, children=[
                            html.Div(style={"margin-right": "30px"}, children=[
                                html.B(series_summary['season_count']), " seasons, ", html.B(series_summary['episode_count']), " episodes, ", 
                                html.B(series_summary['scene_count']), " scenes, ", html.B(series_summary['line_count']), " lines, ", html.B(series_summary['word_count']), " words",
                            ]),
                        ]),
                        html.Br(),
                        html.Div(dcc.Graph(id="series-episodes-scatter-grid")),
                        dcc.RadioItems(
                            id="scatter-grid-hilite",
                            className="text-white", 
                            options=[
                                {'label': 'focal character', 'value': 'focal_speakers'},
                                {'label': 'focal location', 'value': 'focal_locations'},
                                {'label': 'genre (raw)', 'value': 'topics_universal'},
                                {'label': 'genre (weighted by frequency)', 'value': 'topics_universal_tfidf'},
                            ],
                            value='tfidf_score',
                            inputStyle={"margin-left": "12px", "margin-right": "4px"},
                            style={"display": "flex", "padding-bottom": "0"}
                        ),
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
                        html.Br(),
                        dbc.Row([ 
                            dbc.Col(md=12, children=[
                                html.Img(src=wordcloud_img, width="100%")
                            ]),
                        ]),
                    ]),
                ]),
                html.Br(),
                # season episode listing accordion
                dbc.Accordion(id="accordion", active_item="acc_textarea", children=season_accordion_items),
                html.Div(id="accordion-contents", className="mt-3"),
            ]),

            # series continuity timelines for character / location / topic / search
            dbc.CardBody([
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
                            dbc.Tab(label="Topics", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="series-topics-gantt"),
                                ]),
                            ]),
                            dbc.Tab(label="Search", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    html.H3(children=["Search results gantt chart visualization for query \"", html.Span(id='series-dialog-qt-display'), "\""]),
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            "Query term: ",
                                            html.Br(),
                                            dcc.Input(
                                                id="series-dialog-qt",
                                                type="text",
                                                placeholder="enter text to search",
                                                size=30,
                                                autoFocus=True,
                                                debounce=True,
                                                # required=True,
                                            )
                                        ]),
                                    ]),
                                    # NOTE: I believe this button is a placebo: it's a call to action, but simply exiting the qt field invokes the callback 
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            html.Br(),
                                            html.Button(
                                                'Search', 
                                                id='qt-submit',
                                            ),
                                        ]),
                                    ]),
                                ]),
                                html.Br(),
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="series-search-results-gantt-new"),
                                    html.Br(),
                                    html.Div(id="series-search-results-dt"),
                                ]),
                            ]),
                        ]),
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

            # series speaker listing
            dbc.CardBody([
                html.H3("Regular and recurring characters in series"),
                dbc.Row([
                    dbc.Col(md=12, children=[
                        html.Div(id="series-speaker-listing-dt"),
                    ]),
                ]),
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

            # series-speaker-topic mappings
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=12, children=[
                        dbc.Tabs(className="nav nav-tabs", children=[
                            dbc.Tab(label="Episode genres", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            "Topic grouping: ", dcc.Dropdown(id="topic-grouping", options=['universalGenres', 'universalGenresGpt35_v2'], value='universalGenres')
                                        ]),
                                    ]),
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            "Score type: ", dcc.Dropdown(id="score-type", options=['score', 'tfidf_score'], value='tfidf_score')
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
                                            "List episodes for topic: ", dcc.Dropdown(id="parent-topic", options=universal_genres_parent_topics)
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
        ])
    ])

    return content












######################################## SERIES CALLBACKS ##########################################

############ series summary callbacks
@callback(
    Output("accordion-contents", "children"),
    # Output('series-topics', 'children'),
    Input('show-key', 'value'),
    Input("accordion", "active_item"))    
def render_series_summary(show_key: str, expanded_season: str):
    utils.hilite_in_logs(f'callback invoked: render_series_summary, show_key={show_key} expanded_season={expanded_season}')

    # TODO circle back to whether this is needed and how to label it
    # accordion_contents = {}

    return {}


############ all series episodes scatter
@callback(
    Output('series-episodes-scatter-grid', 'figure'),
    Input('show-key', 'value'),
    Input('scatter-grid-hilite', 'value'))    
def render_all_series_episodes_scatter(show_key: str, hilite: str):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_all_series_episodes_scatter ts={callback_start_ts} show_key={show_key} hilite={hilite}')

    if hilite in ['topics_universal', 'topics_universal_tfidf']:
        hilite_color_map = cm.TOPIC_COLORS
    elif hilite == 'focal_speakers':
        speakers = list(show_metadata[show_key]['regular_cast'].keys()) + list(show_metadata[show_key]['recurring_cast'].keys())
        hilite_color_map = cm.generate_speaker_color_discrete_map(show_key, speakers)
    elif hilite == 'focal_locations':
        scenes_by_location_response = esr.agg_scenes_by_location(ShowKey(show_key))
        scenes_by_location = scenes_by_location_response['scenes_by_location']
        locations = utils.truncate_dict(scenes_by_location, 500, start_index=1)
        hilite_color_map = {loc:cm.colors[i % 10] for i, loc in enumerate(locations)}
    else:
        hilite_color_map = None

    season_response = esr.list_seasons(ShowKey(show_key))
    seasons = season_response['seasons']
        
    simple_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
    all_episodes = simple_episodes_response['episodes']
    # all_episodes_dict = {episode['episode_key']:episode for episode in all_episodes}
    # all_episodes = list(all_episodes_dict.values())

    # load all episodes into dataframe
    df = pd.DataFrame(all_episodes)
    df['air_date'] = df['air_date'].apply(lambda x: x[:10])

    cols_to_keep = ['episode_key', 'title', 'season', 'sequence_in_season', 'air_date', 'focal_speakers', 'focal_locations', 
                    'topics_universal', 'topics_universal_tfidf']

    df = df[cols_to_keep]

    all_series_episodes_scatter = pscat.build_all_series_episodes_scatter(df, seasons, hilite=hilite, hilite_color_map=hilite_color_map)

    callback_end_ts = dt.now()
    display_page_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_all_series_episodes_scatter returned at ts={callback_end_ts} duration={display_page_duration}')

    return all_series_episodes_scatter


############ series speakers gantt callback
@callback(
    Output('series-speakers-gantt', 'figure'),
    Input('show-key', 'value'))    
def render_series_speakers_gantt(show_key: str):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_speakers_gantt ts={callback_start_ts} show_key={show_key}')

    episodes_by_season_response = esr.list_simple_episodes_by_season(ShowKey(show_key))
    season_interval_data = gh.simple_season_episode_i_map(episodes_by_season_response['episodes_by_season'])

    file_path = f'./app/data/speaker_gantt_sequence_{show_key}.csv'
    if os.path.isfile(file_path):
        speaker_gantt_sequence_df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        print(f'no file found at file_path={file_path}, running `/esr/generate_series_speaker_gantt_sequence/{show_key}?overwrite_file=True` to generate')
        esr.generate_series_speaker_gantt_sequence(ShowKey(show_key), overwrite_file=True, limit_cast=True)
        if os.path.isfile(file_path):
            speaker_gantt_sequence_df = pd.read_csv(file_path)
            print(f'loading dataframe at file_path={file_path}')
        else:
            raise Exception(f'Failure to render_series_speakers_gantt: unable to fetch or generate dataframe at file_path={file_path}')

    series_speakers_gantt = pgantt.build_series_gantt(show_key, speaker_gantt_sequence_df, 'speakers', interval_data=season_interval_data)

    callback_end_ts = dt.now()
    display_page_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_speakers_gantt returned at ts={callback_end_ts} duration={display_page_duration}')

    return series_speakers_gantt


############ series locations gantt callback
@callback(
    Output('series-locations-gantt', 'figure'),
    Input('show-key', 'value'))    
def render_series_locations_gantt(show_key: str):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_locations_gantt ts={callback_start_ts} show_key={show_key}')

    episodes_by_season_response = esr.list_simple_episodes_by_season(ShowKey(show_key))
    season_interval_data = gh.simple_season_episode_i_map(episodes_by_season_response['episodes_by_season'])

    file_path = f'./app/data/location_gantt_sequence_{show_key}.csv'
    if os.path.isfile(file_path):
        location_gantt_sequence_df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        print(f'no file found at file_path={file_path}, running `/esr/generate_series_location_gantt_sequence/{show_key}?overwrite_file=True` to generate')
        esr.generate_series_location_gantt_sequence(ShowKey(show_key), overwrite_file=True)
        if os.path.isfile(file_path):
            location_gantt_sequence_df = pd.read_csv(file_path)
            print(f'loading dataframe at file_path={file_path}')
        else:
            raise Exception(f'Failure to render_series_locations_gantt: unable to fetch or generate dataframe at file_path={file_path}')
        
    series_locations_gantt = pgantt.build_series_gantt(show_key, location_gantt_sequence_df, 'locations', interval_data=season_interval_data)

    callback_end_ts = dt.now()
    display_page_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_locations_gantt returned at ts={callback_end_ts} duration={display_page_duration}')

    return series_locations_gantt


############ series topics gantt callback
@callback(
    Output('series-topics-gantt', 'figure'),
    Input('show-key', 'value'))    
def render_series_topics_gantt(show_key: str):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_topics_gantt ts={callback_start_ts} show_key={show_key}')

    episodes_by_season_response = esr.list_simple_episodes_by_season(ShowKey(show_key))
    season_interval_data = gh.simple_season_episode_i_map(episodes_by_season_response['episodes_by_season'])

    topic_grouping = 'universalGenres'
    # topic_grouping = f'focusedGpt35_{show_key}'
    topic_threshold = 20
    model_vendor= 'openai'
    model_version = 'ada002'
    file_path = f'./app/data/topic_gantt_sequence_{show_key}_{topic_grouping}_{model_vendor}_{model_version}.csv'
    if os.path.isfile(file_path):
        topic_gantt_sequence_df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        print(f'no file found at file_path={file_path}, running `/esr/generate_series_topic_gantt_sequence/{show_key}?overwrite_file=True` to generate')
        esr.generate_series_topic_gantt_sequence(ShowKey(show_key), overwrite_file=True, topic_grouping=topic_grouping, topic_threshold=topic_threshold,
                                                 model_vendor=model_vendor, model_version=model_version)
        if os.path.isfile(file_path):
            topic_gantt_sequence_df = pd.read_csv(file_path)
            print(f'loading dataframe at file_path={file_path}')
        else:
            raise Exception(f'Failure to render_series_gantts: unable to fetch or generate dataframe at file_path={file_path}')
        
    series_topics_gantt = pgantt.build_series_gantt(show_key, topic_gantt_sequence_df, 'topics', interval_data=season_interval_data)

    callback_end_ts = dt.now()
    display_page_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_topics_gantt returned at ts={callback_end_ts} duration={display_page_duration}')

    return series_topics_gantt


############ series search gantt callback
@callback(
    Output('series-dialog-qt-display', 'children'),
    Output('series-search-results-gantt-new', 'figure'),
    Output('series-search-results-dt', 'children'),
    Input('show-key', 'value'),
    Input('series-dialog-qt', 'value'),
    # NOTE: I believe 'qt-submit' is a placebo: it's a call to action, but simply exiting the qt field invokes the callback
    Input('qt-submit', 'value'))    
def render_series_search_gantt(show_key: str, series_dialog_qt: str, qt_submit: bool = False):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_search_gantt ts={callback_start_ts} show_key={show_key} series_dialog_qt={series_dialog_qt} qt_submit={qt_submit}')

    # execute search query and filter response into series gantt charts

    # TODO fetch from file, but file has to have all speaker data
    # file_path = f'./app/data/speaker_gantt_sequence_{show_key}.csv'
    # if os.path.isfile(file_path):
    #     speaker_gantt_sequence_df = pd.read_csv(file_path)
    #     print(f'loading dataframe at file_path={file_path}')
    # else:
    #     print(f'no file found at file_path={file_path}, running `/esr/generate_series_speaker_gantt_sequence/{show_key}?overwrite_file=True` to generate')
    #     esr.generate_series_speaker_gantt_sequence(ShowKey(show_key), overwrite_file=True)
    #     if os.path.isfile(file_path):
    #         speaker_gantt_sequence_df = pd.read_csv(file_path)
    #         print(f'loading dataframe at file_path={file_path}')
    #     else:
    #         raise Exception('Failure to render_series_gantt_chart: unable to fetch or generate dataframe at file_path={file_path}')
    
    # search_response = esr.search_scene_events(ShowKey(show_key), dialog=qt)
    # series_search_results_gantt = pgantt.build_series_search_results_gantt(show_key, qt, search_response['matches'], speaker_gantt_sequence_df)

    if series_dialog_qt:
        series_gantt_response = esr.generate_series_speaker_gantt_sequence(ShowKey(show_key))
        search_response = esr.search_scene_events(ShowKey(show_key), dialog=series_dialog_qt)
        # if 'matches' not in search_response or len(search_response['matches']) == 0:
        #     print(f"no matches for show_key={show_key} qt=`{qt}` qt_submit=`{qt_submit}`")
        #     return None, show_key, qt
        # print(f"len(search_response['matches'])={len(search_response['matches'])}")
        # print(f"len(series_gantt_response['episode_speakers_sequence'])={len(series_gantt_response['episode_speakers_sequence'])}")
        timeline_df = pd.DataFrame(series_gantt_response['episode_speakers_sequence'])
        series_search_results_gantt = pgantt.build_series_search_results_gantt(show_key, timeline_df, search_response['matches'])

        matching_lines_df = timeline_df.loc[timeline_df['matching_line_count'] > 0]

        # build dash datatable
        matching_lines_df.rename(columns={'Task': 'character', 'sequence_in_season': 'episode'}, inplace=True)
        matching_speakers = matching_lines_df['character'].unique()
        speaker_color_map = cm.generate_speaker_color_discrete_map(show_key, matching_speakers)
        # TODO matching_lines_df['dialog'] = matching_lines_df['dialog'].apply(convert_markup)
        display_cols = ['episode_key', 'episode_title', 'count', 'season', 'episode', 'info', 'matching_line_count', 'matching_lines']
        episode_search_results_dt = cmp.pandas_df_to_dash_dt(matching_lines_df, display_cols, 'episode_key', matching_speakers, speaker_color_map, 
                                                            numeric_precision_overrides={'count': 0, 'season': 0, 'episode': 0, 'matching_line_count': 0})

        # out_text = f"{scene_event_count} lines matching query '{qt}'"

    else:
        series_search_results_gantt = {}
        episode_search_results_dt = {}

    callback_end_ts = dt.now()
    display_page_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_search_gantt returned at ts={callback_end_ts} duration={display_page_duration}')

    return series_dialog_qt, series_search_results_gantt, episode_search_results_dt


############ speaker frequency bar chart callback
@callback(
    Output('speaker-season-frequency-bar-chart', 'figure'),
    Output('speaker-episode-frequency-bar-chart', 'figure'),
    Input('show-key', 'value'),
    Input('speaker-chatter-tally-by', 'value'),
    Input('speaker-chatter-season', 'value'),
    Input('speaker-chatter-sequence-in-season', 'value'))    
def render_speaker_frequency_bar_chart(show_key: str, tally_by: str, season: str, sequence_in_season: str = None):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_speaker_frequency_bar_chart ts={callback_start_ts} show_key={show_key} tally_by={tally_by} season={season} sequence_in_season={sequence_in_season}')

    if season in ['0', 0, 'All']:
        season = None
    else:
        season = int(season)

    # fetch or generate aggregate speaker data and build speaker frequency bar chart
    file_path = f'./app/data/speaker_episode_aggs_{show_key}.csv'
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        print(f'no file found at file_path={file_path}, running `/esr/generate_speaker_line_chart_sequences/{show_key}?overwrite_file=True` to generate')
        esr.generate_speaker_line_chart_sequences(ShowKey(show_key), overwrite_file=True)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            print(f'loading dataframe at file_path={file_path}')
        else:
            raise Exception('Failure to render_speaker_frequency_bar_chart: unable to fetch or generate dataframe at file_path={file_path}')
    
    speaker_season_frequency_bar_chart = pbar.build_speaker_frequency_bar(show_key, df, tally_by, aggregate_ratio=False, season=season)
    speaker_episode_frequency_bar_chart = pbar.build_speaker_frequency_bar(show_key, df, tally_by, aggregate_ratio=False, season=season, sequence_in_season=sequence_in_season)

    callback_end_ts = dt.now()
    display_page_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_speaker_frequency_bar_chart returned at ts={callback_end_ts} duration={display_page_duration}')

    return speaker_season_frequency_bar_chart, speaker_episode_frequency_bar_chart


############ series speaker listing callback
@callback(
    # Output('speaker-qt-display', 'children'),
    Output('series-speaker-listing-dt', 'children'),
    # Output('speaker-matches-dt', 'children'),
    Input('show-key', 'value'))
    # Input('speaker-qt', 'value')) 
def render_series_speaker_listing_dt(show_key: str):
    utils.hilite_in_logs(f'callback invoked: render_series_speaker_listing_dt, show_key={show_key}')   
# def render_series_speaker_listing_dt(show_key: str, speaker_qt: str):
#     print(f'in render_series_speaker_listing_dt, show_key={show_key} speaker_qt={speaker_qt}')
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_speaker_listing_dt ts={callback_start_ts} show_key={show_key}')

    series_speaker_names = list(show_metadata[show_key]['regular_cast'].keys()) + list(show_metadata[show_key]['recurring_cast'].keys())

    indexed_speakers_response = esr.fetch_indexed_speakers(ShowKey(show_key), speakers=','.join(series_speaker_names), extra_fields='topics_mbti')
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

    callback_end_ts = dt.now()
    display_page_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_speaker_listing_dt returned at ts={callback_end_ts} duration={display_page_duration}')

    # return speaker_qt, speaker_listing_dt, speaker_matches_dt
    return speaker_listing_dt


############ series speaker topic grid callback
@callback(
    Output('series-speaker-mbti-scatter', 'figure'),
    Output('series-speaker-dnda-scatter', 'figure'),
    Output('series-speaker-mbti-dt', 'children'),
    Output('series-speaker-dnda-dt', 'children'),
    Input('show-key', 'value'),
    Input('series-mbti-count', 'value'),
    Input('series-dnda-count', 'value'))    
def render_series_speaker_topic_scatter(show_key: str, mbti_count: int, dnda_count: int):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_speaker_topic_scatter ts={callback_start_ts} show_key={show_key} mbti_count={mbti_count} dnda_count={dnda_count}')

    series_speaker_names = list(show_metadata[show_key]['regular_cast'].keys()) + list(show_metadata[show_key]['recurring_cast'].keys())
    indexed_speakers_response = esr.fetch_indexed_speakers(ShowKey(show_key), extra_fields='topics_mbti,topics_dnda', speakers=','.join(series_speaker_names))
    indexed_speakers = indexed_speakers_response['speakers']
    # indexed_speakers = fh.flatten_speaker_topics(indexed_speakers, 'mbti', limit_per_speaker=3) 

    speaker_color_map = cm.generate_speaker_color_discrete_map(show_key, series_speaker_names)

    # flatten episode speaker topic data for each episode speaker
    exploded_speakers_mbti = fflat.explode_speaker_topics(indexed_speakers, 'mbti', limit_per_speaker=mbti_count)
    exploded_speakers_dnda = fflat.explode_speaker_topics(indexed_speakers, 'dnda', limit_per_speaker=dnda_count)
    mbti_df = pd.DataFrame(exploded_speakers_mbti)
    dnda_df = pd.DataFrame(exploded_speakers_dnda)
    series_speaker_mbti_scatter = pscat.build_speaker_topic_scatter(show_key, mbti_df.copy(), 'mbti', speaker_color_map=speaker_color_map)
    series_speaker_dnda_scatter = pscat.build_speaker_topic_scatter(show_key, dnda_df.copy(), 'dnda', speaker_color_map=speaker_color_map)

    # build dash datatable
    display_cols = ['speaker', 'topic_key', 'topic_name', 'score', 'raw_score']
    series_speaker_mbti_dt = cmp.pandas_df_to_dash_dt(mbti_df, display_cols, 'speaker', series_speaker_names, speaker_color_map)
    series_speaker_dnda_dt = cmp.pandas_df_to_dash_dt(dnda_df, display_cols, 'speaker', series_speaker_names, speaker_color_map)

    callback_end_ts = dt.now()
    display_page_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_speaker_topic_scatter returned at ts={callback_end_ts} duration={display_page_duration}')

    return series_speaker_mbti_scatter, series_speaker_dnda_scatter, series_speaker_mbti_dt, series_speaker_dnda_dt


############ series topic pie and bar chart callback
@callback(
    Output('series-topic-pie', 'figure'),
    Output('series-parent-topic-pie', 'figure'),
    Input('show-key', 'value'),
    Input('topic-grouping', 'value'),
    Input('score-type', 'value'))    
def render_series_topic_pies(show_key: str, topic_grouping: str, score_type: str):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_topic_pies ts={callback_start_ts} show_key={show_key} topic_grouping={topic_grouping} score_type={score_type}')

    ##### TODO begin optimization block 
    episode_response = esr.fetch_simple_episodes(ShowKey(show_key))
    episode_topic_lists = []
    for episode in episode_response['episodes']:
        episode_topics_response = esr.fetch_episode_topics(ShowKey(show_key), episode['episode_key'], topic_grouping)
        episode_topic_lists.append(episode_topics_response['episode_topics'])

    series_topics_df, series_parent_topics_df = fflat.generate_topic_aggs_from_episode_topics(episode_topic_lists, max_rank=20, max_parent_repeats=2)
    ##### TODO end optimization block 

    series_topics_pie = ppie.build_topic_aggs_pie(series_topics_df, topic_grouping, score_type)
    series_parent_topics_pie = ppie.build_topic_aggs_pie(series_parent_topics_df, topic_grouping, score_type, is_parent=True)

    callback_end_ts = dt.now()
    display_page_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_topic_pies returned at ts={callback_end_ts} duration={display_page_duration}')

    return series_topics_pie, series_parent_topics_pie


############ series topic episode datatable callback
@callback(
    Output('series-topic-episodes-dt', 'children'),
    Input('show-key', 'value'),
    Input('topic-grouping', 'value'),
    Input('parent-topic', 'value'),
    Input('score-type', 'value'))    
def render_series_topic_episodes_dt(show_key: str, topic_grouping: str, parent_topic: str, score_type: str):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_topic_episodes_dt ts={callback_start_ts} show_key={show_key} topic_grouping={topic_grouping} parent_topic={parent_topic} score_type={score_type}')

    # configurable score threshold
    min_score = 0.5

    # NOTE assembling entire parent-child topic hierarchy here, but only using one branch of the tree
    child_topics = []
    topic_grouping_response = esr.fetch_topic_grouping(topic_grouping)
    for t in topic_grouping_response['topics']:
        # only process topics that have parents (ignore the parents themselves)
        if not t['parent_key']:
            continue
        if parent_topic == t['topic_key'].split('.')[0]:
            child_topics.append(t['topic_key'])

    if not child_topics:
        utils.hilite_in_logs(f'Failure to render_series_topic_episodes_dt, no child topics for parent_topic={parent_topic} in topic_grouping={topic_grouping}')
        return {}

    columns = ['topic_key', 'parent_topic', 'episode_key', 'episode_title', 'season', 'sequence_in_season', 'air_date', 'score', 'tfidf_score']
    topic_episodes_df = pd.DataFrame(columns=columns)
    # for parent_topic, child_topics in parent_to_leaf_topics.items():
    for topic in child_topics:
        episodes_by_topic = esr.find_episodes_by_topic(ShowKey(show_key), topic_grouping, topic)
        df = pd.DataFrame(episodes_by_topic['episode_topics'])
        df['parent_topic'] = parent_topic
        df = df[columns]
        df = df[(df['score'] > min_score) | (df['tfidf_score'] > min_score)]
        topic_episodes_df = pd.concat([topic_episodes_df, df])

    topic_episodes_df.sort_values(score_type, ascending=False, inplace=True)

    series_topic_episodes_dt = cmp.pandas_df_to_dash_dt(topic_episodes_df, columns, 'parent_topic', [parent_topic], cm.TOPIC_COLORS)

    callback_end_ts = dt.now()
    display_page_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_topic_episodes_dt returned at ts={callback_end_ts} duration={display_page_duration}')

    return series_topic_episodes_dt


############ series cluster scatter callback
@callback(
    Output('series-episodes-cluster-scatter', 'figure'),
    Output('series-episodes-cluster-dt', 'children'),
    Input('show-key', 'value'),
    Input('num-clusters', 'value'))    
def render_series_cluster_scatter(show_key: str, num_clusters: int):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_cluster_scatter ts={callback_start_ts} show_key={show_key} num_clusters={num_clusters}')

    num_clusters = int(num_clusters)
    vector_field = 'openai_ada002_embeddings'

    # fetch embeddings for all show episodes 
    s = esqb.fetch_series_embeddings(show_key, vector_field)
    doc_embeddings = esrt.return_all_embeddings(s, vector_field)

    # generate and color-stamp clusters for all show episodes 
    doc_embeddings_clusters_df = ef.cluster_docs(doc_embeddings, num_clusters)
    doc_embeddings_clusters_df['cluster_color'] = doc_embeddings_clusters_df['cluster'].apply(lambda x: cm.colors[x % 10])

    # fetch basic title/season data for all show episodes 
    all_episodes = esr.fetch_simple_episodes(ShowKey(show_key))
    episodes_df = pd.DataFrame(all_episodes['episodes'])

    # merge basic episode data into cluster data
    episodes_df['doc_id'] = episodes_df['episode_key'].apply(lambda x: f'{show_key}_{x}')
    episode_embeddings_clusters_df = pd.merge(doc_embeddings_clusters_df, episodes_df, on='doc_id', how='outer')

    # generate dash_table div as part of callback output
    episode_clusters_df = episode_embeddings_clusters_df[fm.episode_keep_cols + fm.cluster_cols].copy()
    # TODO this flatten_and_format_cluster_df function is a holdover from a bygone era, figure out where/how to generically handle this
    episode_clusters_df = cmp.flatten_and_format_cluster_df(show_key, episode_clusters_df)
    clusters = [str(c) for c in list(episode_clusters_df['cluster'].unique())]
    bg_color_map = {str(i):color for i, color in enumerate(cm.colors)}
    episode_clusters_dt = cmp.pandas_df_to_dash_dt(episode_clusters_df, list(episode_clusters_df.columns), 'cluster', clusters, bg_color_map,
                                                   numeric_precision_overrides={'season': 0, 'episode': 0, 'scenes': 0, 'episode_key': 0, 'cluster': 0})

    # generate scatterplot
    episode_clusters_scatter = pscat.build_cluster_scatter(episode_embeddings_clusters_df, show_key, num_clusters)

    callback_end_ts = dt.now()
    display_page_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_cluster_scatter returned at ts={callback_end_ts} duration={display_page_duration}')

    return episode_clusters_scatter, episode_clusters_dt
