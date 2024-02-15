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


def merge_and_simplify_df(episode_clusters_df: pd.DataFrame) -> html.Div:
    # reformat columns, sort table
    episode_clusters_df['air_date'] = episode_clusters_df['air_date'].apply(lambda x: x[:10])
    episode_clusters_df['focal_speakers'] = episode_clusters_df['focal_speakers'].apply(lambda x: ", ".join(x))
    episode_clusters_df['focal_locations'] = episode_clusters_df['focal_locations'].apply(lambda x: ", ".join(x))
    episode_clusters_df.sort_values(['cluster', 'season', 'sequence_in_season'], inplace=True)
    # rename columns for display
    episode_clusters_df.rename(columns={'sequence_in_season': 'episode', 'scene_count': 'scenes'}, inplace=True)
    # TODO remove this altogether
    episode_clusters_df.drop('cluster_color', axis=1, inplace=True) 
    # generate table div that can function as an identifiable dash object
    table_div = html.Div([
        dash_table.DataTable(
            data=episode_clusters_df.to_dict("records"),
            columns=[{"id": x, "name": x} for x in episode_clusters_df.columns],
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
            ],
        )
    ])
    return table_div
