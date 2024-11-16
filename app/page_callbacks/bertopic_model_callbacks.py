import ast
from bertopic import BERTopic
from dash import callback, Input, Output
import pandas as pd

from app.app_metadata import BERTOPIC_DATA_DIR, BERTOPIC_MODELS_DIR
import app.fig_builder.plotly_bertopic as pbert
import app.fig_builder.plotly_networkgraph as pgraph
import app.page_builder_service.bertopic_page_service as bps
import app.utils as utils

import app.dash.components as cmp
import app.fig_meta.color_meta as cm


############ bertopic-3d-clusters callbacks
@callback(
    Output('bertopic-model-clusters-scatter', 'figure'),
    Output('episode-narratives-per-cluster-dt', 'children'),
    Output('bertopic-visualize-barchart', 'figure'),
    Output('bertopic-visualize-topics', 'figure'),
    Output('bertopic-visualize-hierarchy', 'figure'),
    Output('bertopic-model-id-display', 'children'),
    Input('show-key', 'value'),
    Input('bertopic-model-id', 'value'))    
def render_bertopic_model_clusters(show_key: str, bertopic_model_id: str):
    print(f'in render_bertopic_model_clusters, show_key={show_key} bertopic_model_id={bertopic_model_id}')

    # load cluster data for bertopic model 
    bertopic_model_docs_df = pd.read_csv(f'{BERTOPIC_DATA_DIR}/{show_key}/{bertopic_model_id}.csv', sep='\t')
    # print(f'bertopic_model_docs_df={bertopic_model_docs_df}')

    bertopic_model_docs_df['cluster_title'] = bertopic_model_docs_df['cluster_title'].apply(utils.truncate)
    # bertopic_model_docs_df['cluster'] = bertopic_model_docs_df['cluster_id'].apply(lambda x: f'cluster_{x}')
    bertopic_model_docs_df['cluster'] = bertopic_model_docs_df['cluster_id'].apply(lambda x: str(x))
    bertopic_model_docs_df['topics'] = bertopic_model_docs_df['topics_universal_tfidf_list'].apply(lambda x: ', '.join(ast.literal_eval(x)))
    bertopic_model_docs_df.rename(columns={'title': 'episode_title', 'sequence_in_season': 'episode', 'scene_count': 'scenes', 'Probability': 'probability'}, inplace=True)
    bertopic_model_docs_df.sort_values(['cluster_id', 'season', 'episode', 'probability'], ascending=[True, True, True, False], inplace=True)

    # generate 3d scatter
    bertopic_model_clusters_scatter = pgraph.build_bertopic_model_3d_scatter(show_key, bertopic_model_id, bertopic_model_docs_df)

    # build dash datatable
    bertopic_model_clusters_dt = bps.generate_bertopic_model_clusters_dt(show_key, bertopic_model_docs_df)

    # generate topic keyword maps and topic graphs
    mmr_bertopic_model = BERTopic.load(f'{BERTOPIC_MODELS_DIR}/{show_key}/{bertopic_model_id}/mmr')
    openai_bertopic_model = BERTopic.load(f'{BERTOPIC_MODELS_DIR}/{show_key}/{bertopic_model_id}/openai')
    bertopic_visualize_barchart = pbert.build_bertopic_visualize_barchart(mmr_bertopic_model)
    bertopic_visualize_topics = pbert.build_bertopic_visualize_topics(openai_bertopic_model)
    bertopic_visualize_hierarchy = pbert.build_bertopic_visualize_hierarchy(openai_bertopic_model)

    # return bertopic_3d_scatter, bertopic_visualize_barchart, bertopic_visualize_topics, bertopic_visualize_hierarchy, bertopic_model_id, dash_dt
    return bertopic_model_clusters_scatter, bertopic_model_clusters_dt, bertopic_visualize_barchart, bertopic_visualize_topics, bertopic_visualize_hierarchy, bertopic_model_id
