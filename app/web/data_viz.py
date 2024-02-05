
import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
# from starlette.responses import StreamingResponse


matplotlib.use('AGG')


colors = ["red", "green", "blue", "orange", "purple", "brown", "yellow", "pink", "black", "tomato"]


def generate_graph_matplotlib(df: pd.DataFrame, matrix, num_clusters: int):
    # init figure
    plt.rcParams['figure.figsize'] = [14.0, 8.0]
    plt.rcParams['figure.autolayout'] = True
    fig = plt.figure()  # make sure to call this, in order to create a new figure

    # draw figure
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    print(f'len(vis_dims2)={len(vis_dims2)} vis_dims2={vis_dims2}')

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    print(f'len(x)={len(x)} x={x}')
    print(f'len(y)={len(y)} y={y}')

    print(f'df.columns={df.columns}')

    for category, color in enumerate(colors[:num_clusters]):
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)

        print(f'category={category}, color={color}')
        print(f'xs={xs}')
        print(f'ys={ys}')

        avg_x = xs.mean()
        avg_y = ys.mean()
        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)

    # for i, row in df.iterrows():
    i = 0
    for doc_id, _ in df['doc_id'].items():
        print(f'i={i} doc_id={doc_id}, vis_dims2[i][0]={vis_dims2[i][0]} vis_dims2[i][0]={vis_dims2[i][1]}')
        plt.annotate(doc_id, (vis_dims2[i][0], vis_dims2[i][1]))
        i += 1

    plt.title("Clusters identified visualized in language 2d using t-SNE")

    # store figure
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    return img_buf



def generate_graph_plotly(df: pd.DataFrame, matrix, num_clusters: int):
    # init figure
    plt.rcParams['figure.figsize'] = [14.0, 8.0]
    plt.rcParams['figure.autolayout'] = True
    fig = plt.figure()  # make sure to call this, in order to create a new figure

    # draw figure
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    print(f'len(vis_dims2)={len(vis_dims2)} vis_dims2={vis_dims2}')

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    print(f'len(x)={len(x)} x={x}')
    print(f'len(y)={len(y)} y={y}')

    print(f'df.columns={df.columns}')

    for category, color in enumerate(colors[:num_clusters]):
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)

        print(f'category={category}, color={color}')
        print(f'xs={xs}')
        print(f'ys={ys}')

        avg_x = xs.mean()
        avg_y = ys.mean()
        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)

    # for i, row in df.iterrows():
    i = 0
    for doc_id, _ in df['doc_id'].items():
        print(f'i={i} doc_id={doc_id}, vis_dims2[i][0]={vis_dims2[i][0]} vis_dims2[i][0]={vis_dims2[i][1]}')
        plt.annotate(doc_id, (vis_dims2[i][0], vis_dims2[i][1]))
        i += 1

    plt.title("Clusters identified visualized in language 2d using t-SNE")

    # store figure
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    return img_buf
