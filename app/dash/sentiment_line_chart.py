import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


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
                        "Emotion ",
                        dcc.Dropdown(
                            id="emotion",
                            options=['Joy', 'Love', 'Empathy', 'Curiosity', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Surprise', 'Confusion'],
                            value='Joy',
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