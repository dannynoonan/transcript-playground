import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


def generate_content(episode_dropdown_options: list, episode_key: str, speaker_dropdown_options: list) -> html.Div:
    content = html.Div([
        navbar,
        dbc.Card(className="bg-dark", children=[
            dbc.CardBody([
                html.H3(children=["Character dialog timeline in episode"]),
                dbc.Row([ 
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Show: ",
                            dcc.Dropdown(
                                id="show-key",
                                options=[
                                    {'label': 'TNG', 'value': 'TNG'},
                                    {'label': 'GoT', 'value': 'GoT'},
                                ], 
                                value='TNG',
                            )
                        ]),
                    ]),
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Episode key: ",
                            dcc.Dropdown(
                                id="episode-key",
                                options=episode_dropdown_options,
                                value=episode_key,
                            )
                        ]),
                    ]),
                ]),
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="episode-dialog-timeline-new"),
                ]),
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="episode-location-timeline-new"),
                ]),
                html.Br(),
            ]),
            dbc.CardBody([
                html.H3(children=["Character sentiment timeline"]),
                dbc.Row([
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Freeze on ",
                            dcc.Dropdown(
                                id="freeze-on",
                                options=[
                                    {'label': 'emotion', 'value': 'emotion'},
                                    {'label': 'speaker', 'value': 'speaker'},
                                ], 
                                value='emotion',
                            )
                        ]),
                    ]),
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Emotion ",
                            dcc.Dropdown(
                                id="emotion",
                                options=['Joy', 'Love', 'Empathy', 'Curiosity', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Surprise', 'Confusion'],
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
                            )
                        ]),
                    ]),
                ]),
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="sentiment-line-chart-new"),
                ]),
                html.Br(),
            ]),
            dbc.CardBody([
                html.H3(children=["Character conversations during episode"]),
                dbc.Row(justify="evenly", children=[
                    dbc.Col(md=6, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="speaker-3d-network-graph-new"),
                        ]),
                    ]),
                    dbc.Col(md=6, children=[
                        html.Div([
                            html.Br(),
                            html.Img(src=f"/static/wordclouds/TNG/TNG_{episode_key}.png", style={"padding-left": "10px", "padding-top": "5px"}),
                        ]),
                    ]),       
                ]),
                html.Br(),
            ]),
            dbc.CardBody([
                html.H3(children=["Character prominence in episode"]),
                dbc.Row([
                    dbc.Col(md=6, children=[
                        dbc.Row([
                            dbc.Col(md=4, children=[
                                html.Div([
                                    "Span granularity: ",
                                    dcc.Dropdown(
                                        id="span-granularity",
                                        options=['scene', 'line', 'word'],
                                        value='line',
                                    )
                                ]),
                            ]),
                        ]),
                        dbc.Row([
                            dbc.Col(md=12, children=[
                                html.Div([
                                    html.Br(),
                                    dcc.Graph(id="speaker-episode-frequency-bar-chart-new"),
                                ]),
                            ]),
                        ]),
                    ]),
                    dbc.Col(md=6, children=[
                        dbc.Row([
                            dbc.Col(md=4, children=[
                                html.Div([
                                    "X Axis: ",
                                    dcc.Dropdown(
                                        id="x-axis",
                                        options=['scene_count', 'line_count', 'word_count'],
                                        value='line_count',
                                    )
                                ]),
                            ]),
                            dbc.Col(md=4, children=[
                                html.Div([
                                    "Y Axis: ",
                                    dcc.Dropdown(
                                        id="y-axis",
                                        options=['scene_count', 'line_count', 'word_count'],
                                        value='scene_count',
                                    )
                                ]),
                            ]),
                        ]),
                        dbc.Row([
                            dbc.Col(md=12, children=[
                                html.Div([
                                    html.Br(),
                                    dcc.Graph(id="speaker-chatter-scatter"),
                                ]),
                            ]),       
                        ]),
                        html.Br(),
                    ]),
                ]), 
            ]),
            dbc.CardBody([
                html.H3(children=["Character personalities during episode"]),
                dbc.Row(justify="evenly", children=[
                    dbc.Col(md=6, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="episode-speaker-mbti-scatter"),
                        ]),
                    ]),
                    dbc.Col(md=6, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="episode-speaker-dnda-scatter"),
                        ]),
                    ]),       
                ]),
                html.Br(),
            ]),
            dbc.CardBody([
                html.H3(children=["Episode topic distributions"]),
                dbc.Row([
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Score type ",
                            dcc.Dropdown(
                                id="topic-score-type",
                                options=['raw_score', 'score', 'tfidf_score'], 
                                value='tfidf_score',
                            )
                        ]),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col(md=6, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="episode-universal-genres-treemap"),
                        ]),
                    ]),
                    dbc.Col(md=6, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="episode-universal-genres-gpt35-v2-treemap"),
                        ]),
                    ]),     
                ]),
                html.Br(),
                # dbc.Row([
                #     dbc.Col(md=6, children=[
                #         html.Div([
                #             html.Br(),
                #             dcc.Graph(id="episode-focused-gpt35-treemap"),
                #         ]),
                #     ]),       
                # ]),
                # html.Br(),
            ]),
            dbc.CardBody([
                html.H3(children=["Similar episodes"]),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="episode-similarity-scatter"),
                ]),
            ]),
        ])
    ])

    return content