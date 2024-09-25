import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
import dash.dash_table.Format as dtf
import pandas as pd

import app.fig_builder.fig_helper as fh
import app.fig_builder.fig_metadata as fm
from app.show_metadata import BGCOLORS_TO_TEXT_COLORS


def generate_navbar(episode_dropdown_options: list, episode: dict) -> dbc.Card:
    navbar = dbc.Card(className="text-white bg-primary", style={"z-index":"2000"}, children=[
        dbc.CardBody([
            dbc.Nav(className="nav nav-pills", children=[
                dbc.NavItem(dbc.NavLink("Transcript Playground", style={"color": "#FFFFFF", "font-size": "16pt"}, href="/tsp_dash_new")),
                # dbc.DropdownMenu(label="Shows", menu_variant="dark", nav=True, children=[
                #     dbc.DropdownMenuItem("TNG", style={"color": "#CCCCCC"}, href='/web/show/TNG', target="_blank"), 
                #     dbc.DropdownMenuItem("GoT", style={"color": "#CCCCCC"}, href='/web/show/GoT', target="_blank"), 
                # ]),
                dbc.NavItem(dbc.NavLink("TNG", style={"color": "#FFFFFF"}, href='/web/show/TNG', external_link=True)),
                dbc.NavItem(dbc.NavLink("Episodes", style={"color": "#CCCCCC"}, href='/web/episode_search/TNG', external_link=True)),
                dbc.NavItem(dbc.NavLink("Characters", style={"color": "#CCCCCC"}, href='/web/character_listing/TNG', external_link=True)),
            ])
        ])
    ])
    return navbar


url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


def pandas_df_to_dash_dt(df: pd.DataFrame, display_cols: list, color_key_col: str, color_keys: list, color_map: dict, sort_by: str = None) -> dash_table.DataTable:
    '''
    Turn pandas dataframe into dash_table.DataTable
    '''
    if sort_by:
        df.sort_values(sort_by, ascending=False, inplace=True)

    # https://dash.plotly.com/datatable/conditional-formatting
    style_data_conditional_list = []
    for v in color_keys:
        sdc = {}
        sdc['if'] = dict(filter_query=f"{{{color_key_col}}} = {v}")
        sdc['backgroundColor'] = color_map[v]
        sdc['color'] = BGCOLORS_TO_TEXT_COLORS[color_map[v]]
        style_data_conditional_list.append(sdc)

    dash_dt = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[
            {
                "id": x,
                "name": x, 
                # "presentation": "markdown",
                "type": "numeric",
                "format": dtf.Format(group=dtf.Group.yes, precision=2, scheme=dtf.Scheme.fixed),
            } 
            for x in display_cols],
        style_header={'backgroundColor': 'white', 'fontWeight': 'bold', 'color': 'black'},
        style_cell={'textAlign': 'left', 'font-size': '10pt'},
        style_data_conditional=style_data_conditional_list,
    )

    return dash_dt


def flatten_and_format_topics_df(df: pd.DataFrame, score_type: str) -> pd.DataFrame:
    '''
    TODO copied after being extracted from another function, not sure where / how this sort of dataframe reformatting should be encapsulated
    '''

    df = df[['topic_key', 'topic_name', 'raw_score', 'score', 'is_parent', 'tfidf_score']]
    df['parent_topic'] = df['topic_key'].apply(fh.extract_parent)
    df = df[df['parent_topic'] != df['topic_key']]
    df['total_score'] = df[score_type].sum()

    return df
