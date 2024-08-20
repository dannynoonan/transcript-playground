import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
import pandas as pd

import app.web.fig_metadata as fm


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


def merge_and_simplify_df(clusters_df: pd.DataFrame) -> html.Div:
    '''
    TODO Holy smackers does this need to be cleaned up. Amazingly it sorta works against two different cluster data sets, but either
    (a) needs to be made more generic or (b) any usage of it must share common column names and data types
    '''
    # reformat columns, sort table
    clusters_df['air_date'] = clusters_df['air_date'].apply(lambda x: x[:10])
    # if 'focal_speakers' in clusters_df.columns:
    #     clusters_df['focal_speakers'] = clusters_df['focal_speakers'].apply(lambda x: ", ".join(x))
    # if 'focal_locations' in clusters_df.columns:
    #     clusters_df['focal_locations'] = clusters_df['focal_locations'].apply(lambda x: ", ".join(x))
    clusters_df.sort_values(['cluster', 'season', 'sequence_in_season'], inplace=True)
    # rename columns for display
    clusters_df.rename(columns={'sequence_in_season': 'episode', 'scene_count': 'scenes'}, inplace=True)
    # TODO remove this altogether
    clusters_df.drop('cluster_color', axis=1, inplace=True) 
    # generate table div that can function as an identifiable dash object
    table_div = html.Div([
        dash_table.DataTable(
            data=clusters_df.to_dict("records"),
            columns=[{"id": x, "name": x} for x in clusters_df.columns],
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold',
                'color': 'black',
            },
            style_cell={
                'textAlign': 'left',
                'font-size': '10pt',
            },
            # style_data={
            #     'backgroundColor': 'black',
            #     'color': 'white',
            # },
            style_data_conditional=[
                {
                    'if': {'filter_query': "{cluster} = 0"},
                    'backgroundColor': fm.colors[0],
                    'color': fm.text_colors[0]
                },
                {
                    'if': {'filter_query': "{cluster} = 1"},
                    'backgroundColor': fm.colors[1],
                    'color': fm.text_colors[1]
                },
                {
                    'if': {'filter_query': "{cluster} = 2"},
                    'backgroundColor': fm.colors[2],
                    'color': fm.text_colors[2]
                },
                {
                    'if': {'filter_query': "{cluster} = 3"},
                    'backgroundColor': fm.colors[3],
                    'color': fm.text_colors[3]
                },
                {
                    'if': {'filter_query': "{cluster} = 4"},
                    'backgroundColor': fm.colors[4],
                    'color': fm.text_colors[4]
                },
                {
                    'if': {'filter_query': "{cluster} = 5"},
                    'backgroundColor': fm.colors[5],
                    'color': fm.text_colors[5]
                },
                {
                    'if': {'filter_query': "{cluster} = 6"},
                    'backgroundColor': fm.colors[6],
                    'color': fm.text_colors[6]
                },
                {
                    'if': {'filter_query': "{cluster} = 7"},
                    'backgroundColor': fm.colors[7],
                    'color': fm.text_colors[7]
                },
                {
                    'if': {'filter_query': "{cluster} = 8"},
                    'backgroundColor': fm.colors[8],
                    'color': fm.text_colors[8]
                },
                {
                    'if': {'filter_query': "{cluster} = 9"},
                    'backgroundColor': fm.colors[9],
                    'color': fm.text_colors[9]
                },
                {
                    'if': {'filter_query': "{cluster} = 10"},
                    'backgroundColor': fm.colors[10],
                    'color': fm.text_colors[10]
                },
                {
                    'if': {'filter_query': "{cluster} = 11"},
                    'backgroundColor': fm.colors[11],
                    'color': fm.text_colors[11]
                },
                {
                    'if': {'filter_query': "{cluster} = 12"},
                    'backgroundColor': fm.colors[12],
                    'color': fm.text_colors[12]
                },
                {
                    'if': {'filter_query': "{cluster} = 13"},
                    'backgroundColor': fm.colors[13],
                    'color': fm.text_colors[13]
                },
                {
                    'if': {'filter_query': "{cluster} = 14"},
                    'backgroundColor': fm.colors[14],
                    'color': fm.text_colors[14]
                },
                {
                    'if': {'filter_query': "{cluster} = 15"},
                    'backgroundColor': fm.colors[15],
                    'color': fm.text_colors[15]
                },
                {
                    'if': {'filter_query': "{cluster} = 16"},
                    'backgroundColor': fm.colors[16],
                    'color': fm.text_colors[16]
                },
                {
                    'if': {'filter_query': "{cluster} = 17"},
                    'backgroundColor': fm.colors[17],
                    'color': fm.text_colors[17]
                },
                {
                    'if': {'filter_query': "{cluster} = 18"},
                    'backgroundColor': fm.colors[18],
                    'color': fm.text_colors[18]
                },
                {
                    'if': {'filter_query': "{cluster} = 19"},
                    'backgroundColor': fm.colors[19],
                    'color': fm.text_colors[19]
                },
                {
                    'if': {'filter_query': "{cluster} = 20"},
                    'backgroundColor': fm.colors[20],
                    'color': fm.text_colors[20]
                },
                {
                    'if': {'filter_query': "{cluster} = 21"},
                    'backgroundColor': fm.colors[21],
                    'color': fm.text_colors[21]
                },
                {
                    'if': {'filter_query': "{cluster} = 22"},
                    'backgroundColor': fm.colors[22],
                    'color': fm.text_colors[22]
                },
                {
                    'if': {'filter_query': "{cluster} = 23"},
                    'backgroundColor': fm.colors[23],
                    'color': fm.text_colors[23]
                },
                {
                    'if': {'filter_query': "{cluster} = 24"},
                    'backgroundColor': fm.colors[24],
                    'color': fm.text_colors[24]
                },
                {
                    'if': {'filter_query': "{cluster} = 25"},
                    'backgroundColor': fm.colors[25],
                    'color': fm.text_colors[25]
                },
                {
                    'if': {'filter_query': "{cluster} = 26"},
                    'backgroundColor': fm.colors[26],
                    'color': fm.text_colors[26]
                },
                {
                    'if': {'filter_query': "{cluster} = 27"},
                    'backgroundColor': fm.colors[27],
                    'color': fm.text_colors[27]
                },
                {
                    'if': {'filter_query': "{cluster} = 28"},
                    'backgroundColor': fm.colors[28],
                    'color': fm.text_colors[28]
                },
                {
                    'if': {'filter_query': "{cluster} = 29"},
                    'backgroundColor': fm.colors[29],
                    'color': fm.text_colors[29]
                },
                {
                    'if': {'filter_query': "{cluster} = 30"},
                    'backgroundColor': fm.colors[30],
                    'color': fm.text_colors[30]
                },
                {
                    'if': {'filter_query': "{cluster} = 31"},
                    'backgroundColor': fm.colors[31],
                    'color': fm.text_colors[31]
                },
                {
                    'if': {'filter_query': "{cluster} = 32"},
                    'backgroundColor': fm.colors[32],
                    'color': fm.text_colors[32]
                },
                {
                    'if': {'filter_query': "{cluster} = 33"},
                    'backgroundColor': fm.colors[33],
                    'color': fm.text_colors[33]
                },
                {
                    'if': {'filter_query': "{cluster} = 34"},
                    'backgroundColor': fm.colors[34],
                    'color': fm.text_colors[34]
                },
                {
                    'if': {'filter_query': "{cluster} = 35"},
                    'backgroundColor': fm.colors[35],
                    'color': fm.text_colors[35]
                },
                {
                    'if': {'filter_query': "{cluster} = 36"},
                    'backgroundColor': fm.colors[36],
                    'color': fm.text_colors[36]
                },
                {
                    'if': {'filter_query': "{cluster} = 37"},
                    'backgroundColor': fm.colors[37],
                    'color': fm.text_colors[37]
                },
                {
                    'if': {'filter_query': "{cluster} = 38"},
                    'backgroundColor': fm.colors[38],
                    'color': fm.text_colors[38]
                },
                {
                    'if': {'filter_query': "{cluster} = 39"},
                    'backgroundColor': fm.colors[39],
                    'color': fm.text_colors[39]
                },
            ],
        )
    ])
    return table_div
