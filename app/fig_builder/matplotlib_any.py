import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import app.fig_meta.color_meta as cm


matplotlib.use('AGG')


def build_cluster_scatter_matplotlib(df: pd.DataFrame, show_key: str, num_clusters: int, matrix = None):
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

    for category, color in enumerate(cm.colors[:(num_clusters % 10)]):
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
