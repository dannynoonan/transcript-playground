import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


def generate_content(episode_dropdown_options: list, episode_key: str = None) -> html.Div:
    if not episode_key:
        episode_key = '218'

    content = html.Div([
        navbar,
        dbc.Card(className="bg-dark", children=[
            dbc.CardBody([
                dbc.Row([
                    html.H3(children=["Character comparison chart"]),
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
                                options=['PICARD', 'RIKER', 'DATA', 'TROI', 'LAFORGE', 'WORF', 'CRUSHER'],
                                value='PICARD',
                            )
                        ]),
                    ]),
                ]),
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="sentiment-line-chart"),
                ]),
                html.Br(),
            ]),
        ])
    ])

    return content