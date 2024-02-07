# import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd

import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.nlp.embeddings_factory as ef
import app.web.fig_builder as fb
import app.web.fig_metadata as fm


dapp = Dash(__name__,
            external_stylesheets=[dbc.themes.SOLAR],
            requests_pathname_prefix='/tsp_dash/')


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


dapp.layout = html.Div([
    navbar,
    dbc.Card(className="bg-dark", children=[
        dbc.CardBody([
            dbc.Row([
                html.H3(children=["Cluster groupings for ", html.Span(id='show-key-display')]),
                dbc.Col(md=2, children=[
                    html.Div([
                        "Show: ",
                        dcc.Dropdown(
                            id="show-key",
                            options=[
                                {'label': 'TNG', 'value': 'TNG'},
                                {'label': 'GoT', 'value': 'GoT'},
                            ], 
                            value='TNG',
                        )
                    ]),
                ]),
                dbc.Col(md=2, children=[
                    html.Div([
                        "Number of clusters: ",
                        dcc.Dropdown(
                            id="num-clusters",
                            options=[
                                {'label': '2', 'value': '2'},
                                {'label': '3', 'value': '3'},
                                {'label': '4', 'value': '4'},
                                {'label': '5', 'value': '5'},
                                {'label': '6', 'value': '6'},
                                {'label': '7', 'value': '7'},
                                {'label': '8', 'value': '8'},
                                {'label': '9', 'value': '9'},
                                {'label': '10', 'value': '10'},
                            ], 
                            value='5',
                        )
                    ]),
                ]),
            ]),
            html.Br(),
            dbc.Row(justify="evenly", children=[
                dcc.Graph(id="show-cluster-scatter"),
            ]),
            html.Br(),
            html.Div(id="episodes-df-table"),
        ]),
    ])
])


############ voter-weight-electoral-college-bias-page4 callbacks
@dapp.callback(
    Output('show-cluster-scatter', 'figure'),
    Output('show-key-display', 'children'),
    Output('episodes-df-table', 'children'),
    Input('show-key', 'value'),
    Input('num-clusters', 'value'))    
def render_show_cluster_scatter(show_key: str, num_clusters: int):
    print(f'in render_show_cluster_scatter, show_key={show_key} num_clusters={num_clusters}')
    num_clusters = int(num_clusters)
    vector_field = 'openai_ada002_embeddings'

    # fetch embeddings for all show episodes 
    s = esqb.fetch_all_embeddings(show_key, vector_field)
    doc_embeddings = esrt.return_all_embeddings(s, vector_field)

    # generate and color-stamp clusters for all show episodes 
    doc_embeddings_clusters_df = ef.cluster_docs(doc_embeddings, num_clusters)
    doc_embeddings_clusters_df['cluster_color'] = doc_embeddings_clusters_df['cluster'].apply(lambda x: fm.colors[x])

    # fetch basic title/season data for all show episodes 
    s = esqb.list_episodes_by_season(show_key)
    episodes_by_season = esrt.return_episodes_by_season(s)
    all_episodes = []
    for _, episodes in episodes_by_season.items():
        all_episodes.extend(episodes)
    episodes_df = pd.DataFrame(all_episodes)

    # merge basic episode data into cluster data
    episodes_df['doc_id'] = episodes_df['episode_key'].apply(lambda x: f'{show_key}_{x}')
    episode_embeddings_clusters_df = pd.merge(doc_embeddings_clusters_df, episodes_df, on='doc_id', how='outer')

    # generate dash_table div as part of callback output
    episode_clusters_df = episode_embeddings_clusters_df[fm.episode_keep_cols + fm.cluster_cols].copy()
    table_div = merge_and_simplify_df(episode_clusters_df)

    # generate scatterplot
    fig_scatter = fb.generate_graph_plotly(episode_embeddings_clusters_df, show_key, num_clusters)

    return fig_scatter, show_key, table_div


# TODO where should this live? Can/will it be more generic, or limited to this use case?
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


if __name__ == "__main__":
    dapp.run_server(debug=True)
