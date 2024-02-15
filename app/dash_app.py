# import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd

import app.dash.components as cmp
from app.dash import show_cluster_scatter
from app.dash import show_network_graph
import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.nlp.embeddings_factory as ef
import app.web.fig_builder as fb
import app.web.fig_metadata as fm


dapp = Dash(__name__,
            external_stylesheets=[dbc.themes.SOLAR],
            requests_pathname_prefix='/tsp_dash/')

# app layout
dapp.layout = dbc.Container(fluid=True, children=[
    cmp.url_bar_and_content_div,
])


# Index callbacks
@dapp.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'))
def display_page(pathname):
    if pathname == "/tsp_dash/show-cluster-scatter":
        return show_cluster_scatter.content
    elif pathname == "/tsp_dash/show-network-graph":
        return show_network_graph.content


############ show-cluster-scatter callbacks
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
    s = esqb.fetch_show_embeddings(show_key, vector_field)
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
    table_div = cmp.merge_and_simplify_df(episode_clusters_df)

    # generate scatterplot
    fig_scatter = fb.build_cluster_scatter(episode_embeddings_clusters_df, show_key, num_clusters)

    return fig_scatter, show_key, table_div


############ show-network-graph callbacks
@dapp.callback(
    Output('show-network-graph', 'figure'),
    Output('show-key-display2', 'children'),
    Input('show-key', 'value'))    
def render_show_network_graph(show_key: str):
    print(f'in render_show_network_graph, show_key={show_key}')

    # generate scatterplot
    fig_scatter = fb.build_network_graph()

    return fig_scatter, show_key


if __name__ == "__main__":
    dapp.run_server(debug=True)
