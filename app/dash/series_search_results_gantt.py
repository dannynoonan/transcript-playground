import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


content = html.Div([
    navbar,
    dbc.Card(className="bg-dark", children=[
        dbc.CardBody([
            dbc.Row([
                html.H3(children=["Search results gantt chart visualization for query '", html.Span(id='qt-display'), "' in ", html.Span(id='show-key-display9')]),
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
                        "Query term: ",
                        html.Br(),
                        dcc.Input(
                            id="qt",
                            type="text",
                            placeholder="enter text to search",
                            size=30,
                            autoFocus=True,
                            debounce=True,
                            # required=True,
                        )
                    ]),
                ]),
                # NOTE: I believe this button is a placebo: it's a call to action, but simply exiting the qt field invokes the callback 
                dbc.Col(md=2, children=[
                    html.Div([
                        html.Br(),
                        html.Button(
                            'Search', 
                            id='qt-submit',
                        ),
                    ]),
                ]),
            ]),
            html.Br(),
            dbc.Row(justify="evenly", children=[
                dcc.Graph(id="series-search-results-gantt"),
            ]),
            html.Br(),
        ]),
    ])
])