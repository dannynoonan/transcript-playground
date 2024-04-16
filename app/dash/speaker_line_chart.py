import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


content = html.Div([
    navbar,
    dbc.Card(className="bg-dark", children=[
        dbc.CardBody([
            dbc.Row([
                html.H3(children=["Character comparison chart for ", html.Span(id='show-key-display7')]),
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
                        "Span granularity: ",
                        dcc.Dropdown(
                            id="span-granularity",
                            options=['scene', 'episode', 'line', 'word'],
                            value='scene',
                        )
                    ]),
                ]),
                dbc.Col(md=2, children=[
                    html.Div([
                        "Aggregate? ",
                        dcc.Dropdown(
                            id="aggregate-ratio",
                            options=['True', 'False'],
                            value='True',
                        )
                    ]),
                ]),
                dbc.Col(md=2, children=[
                    html.Div([
                        "Season ",
                        dcc.Dropdown(
                            id="season",
                            options=['All', '1', '2', '3', '4', '5', '6' ,'7'],
                            value='All',
                        )
                    ]),
                ]),
            ]),
            html.Br(),
            dbc.Row(justify="evenly", children=[
                dcc.Graph(id="speaker-line-chart"),
            ]),
            html.Br(),
        ]),
    ])
])