import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar
import app.es.es_read_router as esr
from app.show_metadata import ShowKey


def generate_content(episode_dropdown_options: list, episode_key: str = None) -> html.Div:
    if not episode_key:
        episode_key = '218'

    content = html.Div([
        navbar,
        dbc.Card(className="bg-dark", children=[
            dbc.CardBody([
                dbc.Row([
                    html.H3(children=["3D network graph for ", html.Span(id='show-key-display3')]),
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
                    dcc.Graph(id="speaker-3d-network-graph"),
                ]),
                html.Br(),
            ]),
        ])
    ])

    return content
