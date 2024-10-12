import dash_bootstrap_components as dbc
from dash import dcc, html

import app.dash_new.components as cmp


def generate_content(show_key: str, all_seasons: list, series_summary: dict) -> html.Div:
    # generate navbar
    navbar = cmp.generate_navbar(show_key, all_seasons)

    # define content div
    content = html.Div([
        navbar,
        dbc.Card(className="bg-dark", children=[   

            # series summary
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(md=8, children=[
                        html.H3(className="text-white", children=[
                            html.B(series_summary['series_title'])
                        ]),
                        html.H5(className="text-white", style={'display': 'flex'}, children=[
                            html.Div(style={"margin-right": "30px"}, children=[
                                html.B(series_summary['season_count']), " seasons, ", html.B(series_summary['episode_count']), " episodes, ", 
                                html.B(series_summary['scene_count']), " scenes, ", html.B(series_summary['line_count']), " lines, ", html.B(series_summary['word_count']), " words",
                            ]),
                        ]),
                    ]),
                    dbc.Col(md=4, children=[
                        dbc.Row([ 
                            dbc.Col(md=6),
                            dbc.Col(md=6, children=[
                                html.Div([
                                    "Show: ", dcc.Dropdown(id="show-key", options=[show_key], value=show_key)
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]), 

            # series speaker listing
            dbc.CardBody([
                html.H3("Characters in series"),
                dbc.Row([
                    dbc.Col(md=12, children=[
                        html.Div(id="speaker-listing-dt"),
                    ]),
                ]),
            ]),

        ])
    ])

    return content
