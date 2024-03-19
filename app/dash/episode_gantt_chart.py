import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


# TODO this is almost identical to `speaker_3d_network_graph` and should probably be templatized
def generate_content(episode_dropdown_options: list, episode_key: str = None) -> html.Div:
    if not episode_key:
        episode_key = '218'

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
                    dcc.Graph(id="episode-dialog-timeline"),
                ]),
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="episode-location-timeline"),
                ]),
                html.Br(),
            ]),
        ])
    ])

    return content