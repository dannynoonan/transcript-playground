import dash_bootstrap_components as dbc
from dash import dcc, html

import app.dash_new.components as cmp


def generate_content(show_key: str, episode_key: str, all_seasons: list, episode_dropdown_options: list, emotion_dropdown_options: list) -> html.Div:
    navbar = cmp.generate_navbar(all_seasons)

    content = html.Div([
        navbar,
        dbc.Card(className="bg-dark", children=[

            # episode summary / listing dropdown
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=8, children=[
                        html.H3(className="text-white", children=[html.Span(id='episode-title-summary')]),
                        html.H5(className="text-white", style={'display': 'flex'}, children=[
                            html.Div(style={"margin-right": "30px"}, children=[
                                html.B(id='episode-scene-count'), " scenes, ", html.B(id='episode-line-count'), " lines, ", html.B(id='episode-word-count'), " words",
                            ]),
                            html.Div(style={"margin-right": "30px"}, children=[
                                "Focal characters: ", html.Span(id='episode-focal-speakers')
                            ]),
                            html.Div(style={"margin-right": "10px"}, children=[
                                "Genres: ", html.Span(id='episode-topics')
                            ]),
                        ]),
                        # html.H3(className="text-white", children=[
                        #     "Season ", episode['season'], ", Episode ", episode['sequence_in_season'], ": \"", episode['title'], "\" (", episode['air_date'][:10], ")"]),
                        # html.H5(className="text-white", children=[
                        #     html.B(episode['scene_count']), " scenes, ", html.B(episode['line_count']), " lines, ", html.B(episode['word_count']), " words"]),
                        # html.P(className="text-white", children=['<<  Previous episode  |  Next episode  >>']),
                    ]),
                    dbc.Col(md=4, children=[
                        dbc.Row([ 
                            dbc.Col(md=6, children=[
                                html.Div([
                                    "Show: ", dcc.Dropdown(id="show-key", options=[show_key], value=show_key)
                                ]),
                            ]),
                            dbc.Col(md=6, children=[
                                html.Div([
                                    "Episode: ", dcc.Dropdown(id="episode-key", options=episode_dropdown_options, value=episode_key)
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),

            # episode timelines / sentiment / search
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=12, children=[
                        dbc.Tabs(className="nav nav-tabs", children=[
                            dbc.Tab(label="Timeline by dialog", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="episode-dialog-timeline-new"),
                                ]),
                                dbc.Row([
                                    dbc.Col(md=2, style={'textAlign': 'center'}, children=[
                                        dcc.Checklist(
                                            id="show-layers",
                                            # className="text-white", 
                                            options=[
                                                {'label': 'Show scenes / locations', 'value': 'scene_locations'}
                                            ],
                                            value=[],
                                            inputStyle={"margin-left": "4px", "margin-right": "4px"}
                                        )
                                    ]),
                                ]),
                            ]),
                            dbc.Tab(label="Timeline by location", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="episode-location-timeline-new"),
                                ]),
                            ]),
                            dbc.Tab(label="Timeline by sentiment", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            "Emotion ", dcc.Dropdown(id="emotion", options=emotion_dropdown_options, value='Joy')
                                        ]),
                                    ]),
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            # "Character ", dcc.Dropdown(id="speaker", options=speaker_dropdown_options, value='ALL')
                                            "Character ", dcc.Dropdown(id="episode-speakers")
                                        ]),
                                    ]),
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            "Freeze on ", 
                                            dcc.RadioItems(
                                                id="freeze-on",
                                                className="text-white", 
                                                options=[
                                                    {'label': 'emotion', 'value': 'emotion'},
                                                    {'label': 'character', 'value': 'speaker'}
                                                ],
                                                value='emotion',
                                                inputStyle={"margin-left": "12px", "margin-right": "4px"},
                                                style={"display": "flex", "padding-bottom": "0"}
                                            ),
                                        ]),
                                    ]),
                                ]),
                                html.Br(),
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="sentiment-line-chart-new"),
                                ]),
                            ]),
                            dbc.Tab(label="Timeline search", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=4, children=[
                                        html.Div([
                                            "Search in dialog: ", dbc.Input(id="qt", type="text"),
                                        ]),
                                    ]),
                                ]),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col(md=12, children=[
                                        html.P(children=["Results: ", html.Span(id='out-text')]),
                                        html.Div(dcc.Graph(id="episode-search-results-gantt")),
                                        html.Br(),
                                        html.Div(id="episode-search-results-dt"),
                                    ]), 
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),

            # episode characters
            dbc.CardBody([
                html.H3("Characters in episode"),
                dbc.Row([
                    dbc.Col(md=5, children=[
                        html.Div(id="speaker-episode-summary-dt"),
                        html.Br(),
                        html.Div(dcc.Graph(id="speaker-episode-frequency-bar-chart-new")),
                    ]),
                    dbc.Col(md=7, children=[
                        html.Div(dcc.Graph(id="speaker-3d-network-graph-new")),
                        dcc.RadioItems(
                            id="scale-by",
                            className="text-white", 
                            options=['scenes', 'lines', 'words'],
                            value='lines',
                            inputStyle={"margin-left": "12px", "margin-right": "4px"},
                            style={"display": "flex", "padding-bottom": "0"}
                        ),
                    ]),
                ]),
            ]),

            # similar episodes / wordcloud
            dbc.CardBody([
                html.H3("Similar episodes"),
                dbc.Row([
                    dbc.Col(md=8, children=[
                        html.Div(dcc.Graph(id="episode-similarity-scatter")),
                        html.Div(className="text-white", style={"display": "flex", "padding-bottom": "0"}, children=[
                            dcc.RadioItems(
                                id="mlt-type",
                                className="text-white", 
                                options=[
                                    {'label': 'keyword-based', 'value': 'tfidf'},
                                    {'label': 'embeddings-based', 'value': 'openai_embeddings'},
                                ],
                                value='tfidf',
                                inputStyle={"margin-left": "12px", "margin-right": "4px"},
                                style={"display": "flex", "padding-bottom": "0"}
                            ),
                            # TODO align with right side of episode-similarity-scatter
                            dcc.Checklist(
                                id="show-similar-episodes-dt",
                                options=[
                                    {'label': 'Display as table listing', 'value': 'yes'}
                                ],
                                value=[],
                                inputStyle={"text-align": "right", "margin-left": "400px", "margin-right": "4px"}
                            ),
                        ]),
                        html.Div(id="episode-similarity-dt"),
                    ]),
                    dbc.Col(md=4, children=[
                        # html.Div(html.Img(src=f"/static/wordclouds/TNG/TNG_{episode_key}.png", width='100%')),
                        html.Div(html.Img(id='episode-wordcloud-img', width='100%')),
                    ]), 
                ]),
            ]),

            # episode-speaker-topic mappings
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=12, children=[
                        dbc.Tabs(className="nav nav-tabs", children=[
                            dbc.Tab(label="Character MBTI temperaments", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=7, children=[
                                        html.Div(dcc.Graph(id="episode-speaker-mbti-scatter")),
                                        html.Br(),
                                        dbc.Row([
                                            dbc.Col(md=5, style={"text-align": "right", "color": "white"}, children=['Alt temperaments:']),
                                            dbc.Col(md=4, children=[
                                                dcc.Slider(id="episode-mbti-count", min=1, max=4, step=1, value=3),
                                            ]),
                                        ]),
                                    ]),
                                    dbc.Col(md=5, children=[
                                        html.Div(id="episode-speaker-mbti-dt"),
                                    ]),
                                ]),
                            ]),
                            dbc.Tab(label="Character D&D alignments", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=7, children=[
                                        html.Div(dcc.Graph(id="episode-speaker-dnda-scatter")),
                                        html.Br(),
                                        dbc.Row([
                                            dbc.Col(md=5, style={"text-align": "right", "color": "white"}, children=['Alt alignments:']),
                                            dbc.Col(md=4, children=[
                                                dcc.Slider( id="episode-dnda-count", min=1, max=3, step=1, value=2),
                                            ]),
                                        ]),
                                    ]),
                                    dbc.Col(md=5, children=[
                                        html.Div(id="episode-speaker-dnda-dt"),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),

            # episode-topic mappings
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=12, children=[
                        dbc.Tabs(className="nav nav-tabs", children=[
                            dbc.Tab(label="Genre mappings", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=5, children=[
                                        html.Div(id="episode-universal-genres-dt"),
                                    ]),
                                    dbc.Col(md=7, children=[
                                        html.Div(dcc.Graph(id="episode-universal-genres-treemap")),
                                        dcc.RadioItems(
                                            id="universal-genres-score-type",
                                            className="text-white", 
                                            options=[
                                                {'label': 'absolute scoring', 'value': 'scaled_score'},
                                                {'label': 'frequency-based scoring', 'value': 'tfidf_score'},
                                            ],
                                            value='tfidf_score',
                                            inputStyle={"margin-left": "12px", "margin-right": "4px"},
                                            style={"display": "flex", "padding-bottom": "0"}
                                        ),
                                    ]),
                                ]),
                            ]),
                            dbc.Tab(label="Cluster mappings", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row([
                                    dbc.Col(md=5, children=[
                                        html.Div(id="episode-universal-genres-gpt35-v2-dt"),
                                    ]),
                                    dbc.Col(md=7, children=[
                                        html.Div(dcc.Graph(id="episode-universal-genres-gpt35-v2-treemap")),
                                        dcc.RadioItems(
                                            id="universal-genres-gpt35-v2-score-type",
                                            className="text-white", 
                                            options=[
                                                {'label': 'absolute scoring', 'value': 'scaled_score'},
                                                {'label': 'frequency-based scoring', 'value': 'tfidf_score'},
                                            ],
                                            value='tfidf_score',
                                            inputStyle={"margin-left": "12px", "margin-right": "4px"},
                                            style={"display": "flex", "padding-bottom": "0"}
                                        ),
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
