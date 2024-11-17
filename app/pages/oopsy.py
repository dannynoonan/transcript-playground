import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

import app.page_builder_service.page_components as pc


dash.register_page(__name__)


def layout(err_msg: str) -> html.Div:
    content = html.Div([
        # pc.generate_navbar(),
        dbc.Card(className="bg-dark", children=[
            dbc.CardBody([
                html.Br(),
                dbc.Row([
                    dbc.Col(md=3, children=[
                        html.H3(["Malfunction. Need input: ", err_msg])
                    ]),
                ]),
            ]),
        ]),
    ])
    
    return content
