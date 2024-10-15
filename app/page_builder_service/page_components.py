import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
import dash.dash_table.Format as dtf
import pandas as pd

import app.fig_meta.color_meta as cm


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
    numeric_precision = {col:0 for col in display_cols}
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
