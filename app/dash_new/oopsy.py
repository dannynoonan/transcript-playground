import dash_bootstrap_components as dbc
from dash import dcc, html

from app.dash.components import navbar


def generate_content(err_msg: str) -> html.Div:
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