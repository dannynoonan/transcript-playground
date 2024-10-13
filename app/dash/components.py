import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
import pandas as pd

import app.fig_data.fig_metadata as fm
import app.utils as utils


navbar = dbc.Card(className="text-white bg-primary", style={"z-index":"2000"}, children=[
    dbc.CardBody([
        dbc.Nav(className="nav nav-pills", children=[
            dbc.NavItem(dbc.NavLink("Transcript Playground", style={"color": "#FFFFFF", "font-size": "16pt"}, href="/tsp_dash")),
            # dbc.DropdownMenu(label="Shows", menu_variant="dark", nav=True, children=[
            #     dbc.DropdownMenuItem("TNG", style={"color": "#CCCCCC"}, href='/web/show/TNG', target="_blank"), 
            #     dbc.DropdownMenuItem("GoT", style={"color": "#CCCCCC"}, href='/web/show/GoT', target="_blank"), 
            # ]),
            dbc.NavItem(dbc.NavLink("TNG", style={"color": "#FFFFFF"}, href='/web/show/TNG', external_link=True)),
            dbc.NavItem(dbc.NavLink("Episodes", style={"color": "#CCCCCC"}, href='/web/episode_search/TNG', external_link=True)),
            dbc.NavItem(dbc.NavLink("Character", style={"color": "#CCCCCC"}, href='/web/character_listing/TNG', external_link=True)),
        ])
    ])
])


url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


def flatten_and_format_cluster_df(show_key: str, clusters_df: pd.DataFrame) -> pd.DataFrame:
    '''
    TODO Holy smackers does this need to be cleaned up. Amazingly it sorta works against two different cluster data sets, but either
    (a) needs to be made more generic or (b) any usage of it must share common column names and data types
    '''

    # reformat columns, sort table
    clusters_df['air_date'] = clusters_df['air_date'].apply(lambda x: x[:10])
    if 'focal_speakers' in clusters_df.columns:
        clusters_df['focal_speakers'] = clusters_df['focal_speakers'].apply(lambda x: ", ".join(x))
    if 'focal_locations' in clusters_df.columns:
        clusters_df['focal_locations'] = clusters_df['focal_locations'].apply(lambda x: ", ".join(x))
    clusters_df['link'] = clusters_df.apply(lambda x: utils.wrap_title_in_url(show_key, x['episode_key']), axis=1)
    clusters_df.sort_values(['cluster', 'season', 'sequence_in_season'], inplace=True)

    # rename columns for display
    clusters_df.rename(columns={'sequence_in_season': 'episode', 'scene_count': 'scenes'}, inplace=True)

    # TODO stop populating this color column, row color is set within dash datatable using style_data_conditional filter_query
    clusters_df.drop('cluster_color', axis=1, inplace=True) 

    return clusters_df


def pandas_df_to_dash_dt(df: pd.DataFrame, num_groups: int) -> dash_table.DataTable:
    '''
    Turn pandas dataframe into dash_table.DataTable
    '''

    # https://dash.plotly.com/datatable/conditional-formatting
    style_data_conditional_list = []
    for i in range(num_groups):
        sdc = {}
        sdc['if'] = dict(filter_query=f"{{cluster}} = {i}")
        sdc['backgroundColor'] = fm.colors[i % 10]
        sdc['color'] = fm.text_colors[i % 10]
        style_data_conditional_list.append(sdc)

    dash_dt = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"id": x, "name": x, "presentation": "markdown"} for x in df.columns],
        style_header={'backgroundColor': 'white', 'fontWeight': 'bold', 'color': 'black'},
        style_cell={'textAlign': 'left', 'font-size': '10pt'},
        style_data_conditional=style_data_conditional_list,
    )

    return dash_dt
