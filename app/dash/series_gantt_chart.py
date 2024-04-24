import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


content = html.Div([
    navbar,
    dbc.Card(className="bg-dark", children=[
        dbc.CardBody([
            dbc.Row([
                html.H3(children=["Character continuity gantt chart for ", html.Span(id='show-key-display6')]),
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
            ]),
            html.Br(),
            dbc.Row(justify="evenly", children=[
                dcc.Graph(id="series-speaker-gantt"),
            ]),
            html.Br(),
            dbc.Row(justify="evenly", children=[
                dcc.Graph(id="series-location-gantt"),
            ]),
            html.Br(),
            dbc.Row(justify="evenly", children=[
                dcc.Graph(id="series-topic-gantt"),
            ]),
            html.Br(),
        ]),
    ])
])