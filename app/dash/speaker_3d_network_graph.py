import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


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
                            options=[
                                {'label': 'Cause and Effect', 'value': '218'},
                                {'label': 'Ship in a Bottle', 'value': '238'},
                                {'label': 'Frame of Mind', 'value': '247'},
                            ], 
                            value='218',
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
