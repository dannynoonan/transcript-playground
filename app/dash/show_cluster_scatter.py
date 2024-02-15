import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


content = html.Div([
    navbar,
    dbc.Card(className="bg-dark", children=[
        dbc.CardBody([
            dbc.Row([
                html.H3(children=["Cluster groupings for ", html.Span(id='show-key-display')]),
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
                        "Number of clusters: ",
                        dcc.Dropdown(
                            id="num-clusters",
                            options=[
                                {'label': '2', 'value': '2'},
                                {'label': '3', 'value': '3'},
                                {'label': '4', 'value': '4'},
                                {'label': '5', 'value': '5'},
                                {'label': '6', 'value': '6'},
                                {'label': '7', 'value': '7'},
                                {'label': '8', 'value': '8'},
                                {'label': '9', 'value': '9'},
                                {'label': '10', 'value': '10'},
                            ], 
                            value='5',
                        )
                    ]),
                ]),
            ]),
            html.Br(),
            dbc.Row(justify="evenly", children=[
                dcc.Graph(id="show-cluster-scatter"),
            ]),
            html.Br(),
            html.Div(id="episodes-df-table"),
        ]),
    ])
])
