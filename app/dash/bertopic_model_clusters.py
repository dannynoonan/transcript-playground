import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


def generate_content(bertopic_model_id_options: list, bertopic_model_id: str = None) -> html.Div:
    if not bertopic_model_id:
        bertopic_model_id = 'braycurtis_53_0_25'

    content = html.Div([
        navbar,
        dbc.Card(className="bg-dark", children=[
            dbc.CardBody([
                dbc.Row([
                    html.H3(children=["BERTopic clusters for ", html.Span(id='show-key-display11'), " model ", bertopic_model_id]),
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
                            "BERTopic model id: ",
                            dcc.Dropdown(
                                id="bertopic-model-id",
                                options=bertopic_model_id_options, 
                                value=bertopic_model_id,
                            )
                        ]),
                    ]),
                ]),
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="bertopic-model-clusters"),
                ]),
                # html.Br(),
                # html.Div(id="episode-narratives-per-cluster-df"),
            ]),
        ])
    ])

    return content
