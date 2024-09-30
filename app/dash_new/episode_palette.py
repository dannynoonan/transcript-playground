import dash_bootstrap_components as dbc
from dash import dcc, html

import app.dash_new.components as cmp


def generate_content(all_seasons: list, episode_dropdown_options: list, episode: dict, speaker_dropdown_options: list, emotion_dropdown_options: list) -> html.Div:
    navbar = cmp.generate_navbar(all_seasons, episode_dropdown_options, episode)

    content = html.Div([
        navbar,
        dbc.Card(className="bg-dark", children=[
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=8, children=[
                        html.H3(className="text-white", children=[
                            "Season ", episode['season'], ", Episode ", episode['sequence_in_season'], ": \"", episode['title'], "\" (", episode['air_date'][:10], ")"]),
                        html.H5(className="text-white", children=[
                            html.B(episode['scene_count']), " scenes, ", html.B(episode['line_count']), " lines, ", html.B(episode['word_count']), " words"]),
                        # html.P(className="text-white", children=['<<  Previous episode  |  Next episode  >>']),
                    ]),
                    dbc.Col(md=4, children=[
                        dbc.Row([ 
                            dbc.Col(md=6, children=[
                                html.Div([
                                    "Show: ",
                                    dcc.Dropdown(
                                        id="show-key",
                                        options=['TNG', 'GoT'],
                                        value='TNG',
                                    )
                                ]),
                            ]),
                            dbc.Col(md=6, children=[
                                html.Div([
                                    "Episode: ",
                                    dcc.Dropdown(
                                        id="episode-key",
                                        options=episode_dropdown_options,
                                        value=episode['episode_key'],
                                    )
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
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
                                            "Freeze on ",
                                            dcc.Dropdown(
                                                id="freeze-on",
                                                options=['emotion', 'speaker'],
                                                value='emotion',
                                            )
                                        ]),
                                    ]),
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            "Emotion ",
                                            dcc.Dropdown(
                                                id="emotion",
                                                options=emotion_dropdown_options,
                                                value='Joy',
                                            )
                                        ]),
                                    ]),
                                    dbc.Col(md=2, children=[
                                        html.Div([
                                            "Speaker ",
                                            dcc.Dropdown(
                                                id="speaker",
                                                options=speaker_dropdown_options,
                                                value='ALL'
                                            )
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
                                            "Search in dialog: ",
                                            dbc.Input(
                                                id="qt", 
                                                type="text"
                                            ),
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
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=5, children=[
                        html.Div(id="speaker-episode-summary-dt"),
                        html.Br(),
                        html.Div(dcc.Graph(id="speaker-episode-frequency-bar-chart-new")),
                        dcc.RadioItems(
                            id="scale-by",
                            className="text-white", 
                            options=['scenes', 'lines', 'words'],
                            value='lines',
                            inputStyle={"margin-left": "12px", "margin-right": "4px"},
                            style={"display": "flex", "padding-bottom": "0"}
                        ),
                    ]),
                    dbc.Col(md=7, children=[
                        html.Div(dcc.Graph(id="speaker-3d-network-graph-new")),
                    ]),
                ]),
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=8, children=[
                        html.Div([
                            dcc.Graph(id="episode-similarity-scatter"),
                        ]),
                        dcc.RadioItems(
                            id="mlt-type",
                            className="text-white", 
                            options=['tfidf', 'openai_embeddings'],
                            value='tfidf',
                            inputStyle={"margin-left": "12px", "margin-right": "4px"},
                            style={"display": "flex", "padding-bottom": "0"}
                        ),
                    ]),
                    dbc.Col(md=4, children=[
                        html.Div([
                            html.Img(src=f"/static/wordclouds/TNG/TNG_{episode['episode_key']}.png", width='100%',
                                    #  style={"padding-left": "10px", "padding-top": "5px"}
                            ),
                        ]),
                    ]), 
                ]),
            ]),
            dbc.CardBody([
                html.H3("Character personalities during episode"),
                dbc.Row(justify="evenly", children=[
                    dbc.Col(md=6, children=[
                        html.Div([
                            dcc.Graph(id="episode-speaker-mbti-scatter"),
                        ]),
                        html.Br(),
                        html.Div(id="episode-speaker-mbti-dt"),
                    ]),
                    dbc.Col(md=6, children=[
                        html.Div([
                            dcc.Graph(id="episode-speaker-dnda-scatter"),
                        ]),
                        html.Br(),
                        html.Div(id="episode-speaker-dnda-dt"),
                    ]),       
                ]),
            ]),
            dbc.CardBody([
                html.H3("Episode topic distributions"),
                dbc.Row([
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Score type ",
                            dcc.Dropdown(
                                id="topic-score-type",
                                options=['scaled_score', 'tfidf_score'], 
                                value='tfidf_score',
                            )
                        ]),
                    ]),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(md=6, children=[
                        html.Div([
                            dcc.Graph(id="episode-universal-genres-treemap"),
                        ]),
                        html.Br(),
                        html.Div(id="episode-universal-genres-dt"),
                    ]),
                    dbc.Col(md=6, children=[
                        html.Div([
                            dcc.Graph(id="episode-universal-genres-gpt35-v2-treemap"),
                        ]),
                        html.Br(),
                        html.Div(id="episode-universal-genres-gpt35-v2-dt")
                    ]),     
                ]),
            ])
        ])
    ])

    return content
