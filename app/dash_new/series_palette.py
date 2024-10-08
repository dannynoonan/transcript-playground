import dash_bootstrap_components as dbc
from dash import dcc, html

import app.dash_new.components as cmp


def generate_content(show_key: str, all_seasons: list, universal_genres_parent_topics: list) -> html.Div:
    navbar = cmp.generate_navbar(all_seasons)

    content = html.Div([
        navbar,
        dbc.Card(className="bg-dark", children=[

            # series summary
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=8, children=[
                        html.H3(className="text-white", children=[html.B(id='series-title-summary'), " (", html.Span(id='series-air-date-range'), ")"]),
                        html.H5(className="text-white", style={'display': 'flex'}, children=[
                            html.Div(style={"margin-right": "30px"}, children=[
                                html.B(id='series-season-count'), " seasons, ", html.B(id='series-episode-count'), " episodes, ", 
                                html.B(id='series-scene-count'), " scenes, ", html.B(id='series-line-count'), " lines, ", html.B(id='series-word-count'), " words",
                            ]),
                            # html.Div(style={"margin-right": "10px"}, children=[
                            #     "Predominant genres: ", html.Span(id='series-topics')
                            # ]),
                        ]),
                        html.Br(),
                        html.Div(dcc.Graph(id="all-series-episodes-scatter")),
                        html.Div(className="text-white", style={"display": "flex", "padding-bottom": "0"}, children=[
                            dcc.Checklist(
                                id="show-all-series-episodes-dt",
                                options=[
                                    {'label': 'Display as table listing', 'value': 'yes'}
                                ],
                                value=[],
                                inputStyle={"margin-left": "12px", "margin-right": "4px"},
                            ),
                        ]),
                        # html.Div(id="all-series-episodes-dt"),
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
                                html.Div(html.Img(id='series-wordcloud-img', width='100%'))
                            ]),
                        ]),
                    ]),
                ]),
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
                            "Span granularity: ",
                            dcc.Dropdown(
                                id="span-granularity",
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
                                id="character-chatter-season",
                                min=0,
                                max=7,
                                step=None,
                                marks={
                                    int(y): {'label': str(y), 'style': {'transform': 'rotate(45deg)', 'color': 'white'}}
                                    for y in range(0,8)
                                },
                                value=1,
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
                                id="character-chatter-sequence-in-season",
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
                html.H3("Characters in series"),
                dbc.Row([
                    dbc.Col(md=12, children=[
                        html.Div(id="speaker-series-listing-dt"),
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
                            dbc.Tab(label="Series topics", tab_style={"font-size": "20px", "color": "white"}, children=[
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
                            dbc.Tab(label="Episode clustering", tab_style={"font-size": "20px", "color": "white"}, children=[
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
