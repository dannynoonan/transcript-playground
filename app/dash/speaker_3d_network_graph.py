import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar
import app.es.es_read_router as esr
from app.show_metadata import ShowKey


all_simple_episodes = esr.fetch_all_simple_episodes(ShowKey('TNG'))
episode_dropdown_options = []
for episode in all_simple_episodes['episodes']:
    episode_dropdown_options.append({'label': f"{episode['title']} (S{episode['season']}:E{episode['sequence_in_season']})" , 'value': episode['episode_key']})


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
