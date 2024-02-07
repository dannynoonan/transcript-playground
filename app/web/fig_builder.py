import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE

import app.web.fig_metadata as fm


matplotlib.use('AGG')



def generate_graph_matplotlib(df: pd.DataFrame, show_key: str, num_clusters: int, matrix = None):
    # shows how little I understand: sorting here completely alters the plot, but in a way that's deterministic (and consistent with the plotly version)
    df.sort_values(['doc_id'], inplace=True)

    # init figure
    plt.rcParams['figure.figsize'] = [14.0, 8.0]
    plt.rcParams['figure.autolayout'] = True
    fig = plt.figure()

    # draw figure
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    new_df = df.drop(['doc_id', 'cluster'], axis=1)
    vis_dims2 = tsne.fit_transform(new_df)

    x = [x for x, _ in vis_dims2]
    y = [y for _, y in vis_dims2]

    for category, color in enumerate(fm.colors[:num_clusters]):
        xs = np.array(x)[df.cluster == category]
        ys = np.array(y)[df.cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)

        avg_x = xs.mean()
        avg_y = ys.mean()
        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)

    # for i, row in df.iterrows():
    i = 0
    for doc_id, _ in df['doc_id'].items():
        plt.annotate(doc_id, (vis_dims2[i][0], vis_dims2[i][1]))
        i += 1

    plt.title(f'{num_clusters} clusters for {show_key} visualized in 2D using t-SNE')

    # store figure
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    return img_buf


def generate_graph_plotly(episode_embeddings_clusters_df: pd.DataFrame, show_key: str, num_clusters: int) -> go.Figure:
    fig_width = fm.fig_dims.MD11
    fig_height = fm.fig_dims.hdef(fig_width)
    base_fig_title = f'{num_clusters} clusters for {show_key} visualized in 2D using t-SNE'
    custom_data = ['title', 'season', 'sequence_in_season', 'air_date', 'episode_key']
    # hover_data = {'title': True, 'season': True, 'sequence_in_season': True, 'air_date': True, 'episode_key': True}

    # generate dimensional reduction of embeddings
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    doc_embeddings_df = episode_embeddings_clusters_df.drop(fm.episode_drop_cols + fm.episode_keep_cols + fm.cluster_cols, axis=1)
    # doc_embeddings_df.columns = doc_embeddings_df.columns.astype(str)
    vis_dims2 = tsne.fit_transform(doc_embeddings_df)
    x = [x for x, _ in vis_dims2]
    y = [y for _, y in vis_dims2]

    # copy df subset needed for display
    # episode_clusters_df = episode_embeddings_clusters_df[fm.episode_keep_cols + fm.cluster_cols].copy()

    # init figure with core properties
    fig = px.scatter(episode_embeddings_clusters_df, x=x, y=y, 
                     color=episode_embeddings_clusters_df['cluster_color'],
                     color_discrete_map=fm.color_map,
                     title=base_fig_title, custom_data=custom_data,
                     # hover_name=episode_clusters_df.episode_key, hover_data=hover_data,
                     height=fig_height, width=fig_width, opacity=0.7)
    # axis metadata
    fig.update_xaxes(title_text='x axis of t-SNE')
    fig.update_yaxes(title_text='y axis of t-SNE')

    # rollover display data metadata
    fig.update_traces(
        hovertemplate = "<br>".join([
            "<b>%{customdata[0]}</b><br>",
            "Season: <b>%{customdata[1]}</b>",
            "Sequence: <b>%{customdata[2]}</b>",
            "Air date: <b>%{customdata[3]}</b>",
            "Episode key: <b>%{customdata[4]}</b>"
        ])
    )

    return fig
