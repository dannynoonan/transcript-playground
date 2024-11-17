import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime as dt

import app.es.es_read_router as esr
from app.page_callbacks.bertopic_model_callbacks import *
import app.page_builder_service.page_components as pc
from app.show_metadata import ShowKey
from app import utils


dash.register_page(__name__, path_template='/bertopic_model/<show_key>/<bertopic_model_id>')


def layout(show_key: str, bertopic_model_id: str) -> html.Div:

    display_page_start_ts = dt.now()
    utils.hilite_in_logs(f'PAGE LOAD: /bertopic_model/{show_key}/{bertopic_model_id} at ts={display_page_start_ts}')

    ##################### BEGIN FETCH ON PAGE LOAD #####################

    # all seasons
    all_seasons_response = esr.list_seasons(ShowKey(show_key))
    all_seasons = all_seasons_response['seasons']

    # all bertopic models
    bertopic_model_list_response = esr.list_bertopic_models(show_key)
    bertopic_model_options = bertopic_model_list_response['bertopic_model_ids']

    ##################### END FETCH ON PAGE LOAD #####################

    display_page_end_ts = dt.now()
    display_page_duration = display_page_end_ts - display_page_start_ts
    utils.hilite_in_logs(f'LAYOUT: /bertopic_model/{show_key}/{bertopic_model_id} at ts={display_page_end_ts} duration={display_page_duration}')

    navbar = pc.generate_navbar(show_key, all_seasons)

    content = html.Div([
        # page storage
        # TODO

        # page display
        navbar,
        dbc.Card(className="bg-dark", children=[
            dbc.CardBody([
                dbc.Row([
                    html.H3(children=["BERTopic clusters for model_id ", html.Span(id='bertopic-model-id-display')]),
                    # html.H3(children=["BERTopic clusters"]),
                    dbc.Col(md=2, children=[
                        html.Div([
                            "Show: ",
                            dcc.Dropdown(
                                id="show-key",
                                options=[
                                    {'label': 'TNG', 'value': 'TNG'},
                                    # {'label': 'GoT', 'value': 'GoT'},
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
                                options=bertopic_model_options, 
                                value=bertopic_model_id,
                            )
                        ]),
                    ]),
                ]),
                html.Br(),
                dbc.Row(justify="evenly", children=[
                    dcc.Graph(id="bertopic-model-clusters-scatter"),
                ]),
            ]),

            # bertopic model metadata / episode narrative mapping lists
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=12, children=[
                        dbc.Tabs(className="nav nav-tabs", children=[
                            dbc.Tab(label="Episode-narrative listings", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row(justify="evenly", children=[
                                    html.Div(id="episode-narratives-per-cluster-dt"),
                                ]),
                            ]),
                            dbc.Tab(label="Topic word scores", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="bertopic-visualize-barchart"),
                                ]),
                            ]),
                            dbc.Tab(label="Topic distances", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="bertopic-visualize-topics"),
                                ]),
                            ]),
                            dbc.Tab(label="Topic hierarchy", tab_style={"font-size": "20px", "color": "white"}, children=[
                                dbc.Row(justify="evenly", children=[
                                    dcc.Graph(id="bertopic-visualize-hierarchy"),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ])
    ])

    return content
