import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
import dash.dash_table.Format as dtf
import pandas as pd

import app.figdata_manager.color_meta as cm
import app.figdata_manager.matrix_operations as mxop
from app import utils


def generate_navbar(show_key: str, season_dropdown_options: list) -> dbc.Card:

    season_dropdown_menu = []
    for season in season_dropdown_options:
        season_menu_item = dbc.DropdownMenuItem(str(season), style={"color": "White"}, href='/web/show/TNG', target="_blank")
        season_dropdown_menu.append(season_menu_item)

    navbar = dbc.Card(className="text-white bg-primary", style={"z-index":"2000"}, children=[
        dbc.CardBody([
            dbc.Nav(className="nav nav-pills", children=[
                dbc.NavItem(dbc.NavLink("Transcript Playground", style={"color": "White", "font-size": "16pt"}, href="/dash_pages")),
                dbc.DropdownMenu(label="Shows", color="primary", children=[
                    dbc.DropdownMenuItem("TNG", style={"color": "White"}, target="_blank",
                                         href=f'/dash_pages/series/{show_key}'), 
                ]),
                dbc.NavItem(dbc.NavLink(show_key, style={"color": "White"}, external_link=True,
                                        href=f'/dash_pages/series/{show_key}')),
                dbc.DropdownMenu(label="Seasons", color="primary", children=season_dropdown_menu),
                dbc.NavItem(dbc.NavLink("Search", style={"color": "White"}, href='/web/episode_search/TNG', external_link=True)),
                dbc.NavItem(dbc.NavLink("Episodes", style={"color": "White"}, href='/web/episode_search/TNG', external_link=True)),
                dbc.NavItem(dbc.NavLink("Characters", style={"color": "White"}, external_link=True,
                                        href=f'/dash_pages/character_listing/{show_key}')),
                dbc.NavItem(dbc.NavLink("Topics", style={"color": "White"}, href='/web/topic_listing/TNG', external_link=True)),
            ])
        ])
    ])

    return navbar


url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


def pandas_df_to_dash_dt(df: pd.DataFrame, display_cols: list, color_key_col: str, conditional_color_keys: list, bg_color_map: dict, 
                         numeric_precision_overrides: dict = None) -> dash_table.DataTable:
    '''
    Turn pandas dataframe into dash_table.DataTable
    '''

    # fun fact: dash_table 'columns' property doesn't prevent data from being loaded into DataTable, it just limits what's displayed
    df = df[display_cols]

    # kinda gross: default numeric precision is 2, if we want something else then override it in numeric_precision_overrides
    numeric_precision = {col:2 for col in display_cols}
    if numeric_precision_overrides:
        for col, precision in numeric_precision_overrides.items():
            numeric_precision[col] = precision

    # https://dash.plotly.com/datatable/conditional-formatting
    style_data_conditional_list = []
    for i, v in enumerate(conditional_color_keys):
        sdc = {}
        # if color keys are already actual colors, no need to get them from reference map
        if v.startswith('rgb'):
            sdc['if'] = dict(filter_query=f"{{{color_key_col}}} = \"{i+1}\"")
            sdc['backgroundColor'] = v
            sdc['color'] = 'Black'
        else:
            # If v isn't escaped, `if: filter_query` block will cause the page to infinitely reload. Escaping causes the color_map match to fail.
            # TODO circle back to address how to store special chars in these reference sets
            v_escaped = v.replace("'", "") 
            v_escaped = v.replace("\"", "") 
            sdc['if'] = dict(filter_query=f"{{{color_key_col}}} = \"{v_escaped}\"")
            sdc['backgroundColor'] = bg_color_map[v]
            if bg_color_map[v] in cm.BGCOLORS_TO_TEXT_COLORS:
                sdc['color'] = cm.BGCOLORS_TO_TEXT_COLORS[bg_color_map[v]]
            else:
                sdc['color'] = 'Black'
        style_data_conditional_list.append(sdc)

    columns=[
        {
            "id": col, "name": col, "type": "numeric", 
            # "presentation": "markdown",
            "format": dtf.Format(group=dtf.Group.yes, precision=numeric_precision[col], scheme=dtf.Scheme.fixed)
        }
        for col in display_cols]

    dash_dt = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=columns,
        style_header={'backgroundColor': 'white', 'fontWeight': 'bold', 'color': 'black', 'position': 'sticky', 'top': '0'},
        style_cell={'textAlign': 'left', 'font-size': '10pt', 'whiteSpace': 'normal', 'height': 'auto'},
        style_data_conditional=style_data_conditional_list,
        markdown_options={"html": True},
        style_table={'maxHeight': '850px', 'overflowY': 'auto'}
    )

    return dash_dt


# TODO this is an exact copy of flatten_and_format_cluster_df from dash.components
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


def generate_season_episodes_accordion_items(all_season_dicts: dict, speaker_color_map: dict) -> list:
    season_accordion_items = []

    for season, season_dict in all_season_dicts.items():
        # label for collapsed season accordion item
        season_title_text = f"Season {season} ({season_dict['air_date_begin']} — {season_dict['air_date_end']}): {len(season_dict['episodes'])} episodes"

        # episode listing datatable for expanded season accordion item
        season_episodes_dt = generate_season_episodes_dt(season_dict['episodes'])
        
        # recurring speaker datatable
        recurring_speaker_cols = ['character', 'lines']
        recurring_speaker_df = pd.DataFrame(season_dict['speaker_line_counts'].items(), columns=recurring_speaker_cols)
        speaker_list = list(season_dict['speaker_line_counts'].keys())
        recurring_speaker_dt = pandas_df_to_dash_dt(recurring_speaker_df, recurring_speaker_cols, 'character', speaker_list, speaker_color_map,
                                                    numeric_precision_overrides={'lines': 0})

        # recurring location datatable
        recurring_location_cols = ['location', 'scenes']
        recurring_location_df = pd.DataFrame(season_dict['location_counts'].items(), columns=recurring_location_cols)
        locations_list = list(season_dict['location_counts'].keys())
        bg_color_map = {loc:'DarkSlateBlue' for loc in locations_list}
        recurring_location_dt = pandas_df_to_dash_dt(recurring_location_df, recurring_location_cols, 'location', locations_list, bg_color_map, 
                                                     numeric_precision_overrides={'scenes': 0})

        # combine elements into accordion item dash object
        accordion_children = [
            dbc.Row([
                dbc.Col(md=8, children=[season_episodes_dt]),
                dbc.Col(md=2, children=[recurring_speaker_dt]),
                dbc.Col(md=2, children=[recurring_location_dt])
            ])
        ]
        season_accordion_item = dbc.AccordionItem(title=season_title_text, item_id=season, children=accordion_children)
        season_accordion_items.append(season_accordion_item)

    return season_accordion_items


def generate_season_episodes_dt(episodes: list) -> dash_table.DataTable:
    episodes_df = pd.DataFrame(episodes)

    # field naming and processing
    episodes_df['focal_characters'] = episodes_df['focal_speakers'].apply(lambda x: ', '.join(x))
    episodes_df['genres'] = episodes_df.apply(lambda x: mxop.flatten_topics(x['topics_universal_tfidf'], parent_only=True), axis=1)
    episodes_df['air_date'] = episodes_df['air_date'].apply(lambda x: x[:10])
    episodes_df.rename(columns={'sequence_in_season': 'episode'}, inplace=True) 

    # table display input
    display_cols = ['episode', 'title', 'air_date', 'focal_characters', 'genres']
    episode_list = [str(e) for e in list(episodes_df['episode'].unique())]
    bg_color_map = {e:'Maroon' for e in episode_list}

    # convert to dash datatable
    episodes_dt = pandas_df_to_dash_dt(episodes_df, display_cols, 'episode', episode_list, bg_color_map, 
                                       numeric_precision_overrides={'episode': 0})

    return episodes_dt
