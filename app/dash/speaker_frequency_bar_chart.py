import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


content = html.Div([
    navbar,
    dbc.Card(className="bg-dark", children=[
        dbc.CardBody([
            dbc.Row([
                html.H3(children=["Character frequency for ", html.Span(id='show-key-display10')]),
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
                            value='line',
                        )
                    ]),
                ]),
                # dbc.Col(md=2, children=[
                #     html.Div([
                #         "Aggregate? ",
                #         dcc.Dropdown(
                #             id="aggregate-ratio",
                #             options=['True', 'False'],
                #             value='True',
                #         )
                #     ]),
                # ]),
            ]),
            html.Br(),
            dbc.Row(justify="evenly", children=[
                dbc.Col(md=6, children=[
                    html.Div([
                        "Season ",
                        html.Br(),
                        dcc.Slider(
                            id="season",
                            min=0,
                            max=7,
                            step=None,
                            marks={
                                int(y): {'label': str(y), 'style': {'transform': 'rotate(45deg)', 'color': 'white'}}
                                for y in range(0,8)
                            },
                            value=1,
                        ),
                        html.Br(),
                        dcc.Graph(id="speaker-frequency-bar-chart"),
                    ]),
                ]),
                dbc.Col(md=6, children=[
                    html.Div([
                        html.Br(),
                    ]),
                ]),
            ]),
            html.Br(),
        ]),
    ])
])