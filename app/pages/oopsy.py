import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from app.dash.components import navbar


dash.register_page(__name__)


def layout(err_msg: str) -> html.Div:
    content = html.Div([
        navbar,
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