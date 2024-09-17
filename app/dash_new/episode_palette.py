import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


def generate_content(episode_dropdown_options: list, episode_key: str, speaker_dropdown_options: list) -> html.Div:
    content = html.Div([
        navbar,
        dbc.Card(className="bg-dark", children=[
            dbc.CardBody([
                dbc.Row([
                    html.H3(children=["Character dialog gantt chart for ", html.Span(id='show-key-display5')]),
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
                dbc.Row([
                    html.H3(children=["Character comparison chart"]),
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Show: ",
                            dcc.Dropdown(
                                id="show-key2",
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
                                id="episode-key2",
                                options=episode_dropdown_options,
                                value=episode_key,
                            )
                        ]),
                    ]),
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
                dbc.Row([
                    html.H3(children=["3D network graph"]),
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Show: ",
                            dcc.Dropdown(
                                id="show-key3",
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
                                id="episode-key3",
                                options=episode_dropdown_options,
                                value=episode_key,
                            )
                        ]),
                    ]),
                ]),
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="speaker-3d-network-graph-new"),
                ]),
                html.Br(),
            ]),
            dbc.CardBody([
                dbc.Row([
                    html.H3(children=["Character frequency"]),
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Show: ",
                            dcc.Dropdown(
                                id="show-key4",
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
                                id="episode-key4",
                                options=episode_dropdown_options,
                                value=episode_key,
                            )
                        ]),
                    ]),
                    dbc.Col(md=2, children=[
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
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dbc.Col(md=6, children=[
                        html.Div([
                            html.Br(),
                            dcc.Graph(id="speaker-episode-frequency-bar-chart-new"),
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
                dbc.Row([
                    html.H3(children=["Character personalities"]),
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Show: ",
                            dcc.Dropdown(
                                id="show-key5",
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
                                id="episode-key5",
                                options=episode_dropdown_options,
                                value=episode_key,
                            )
                        ]),
                    ]),
                ]),
                html.Br(),
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
        ])
    ])

    return content