import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
# from starlette.responses import StreamingResponse


matplotlib.use('AGG')


colors = ["red", "green", "blue", "orange", "purple", "brown", "yellow", "pink", "black", "tomato"]


class FigDimensions():
    def __init__(self):
        self.MD5 = 650
        self.MD6 = 788
        self.MD7 = 924
        self.MD8 = 1080
        self.MD10 = 1320
        self.MD11 = 1450
        self.MD12 = 1610

    def square(self, width):
        return width

    def hdef(self, width):
        return width * .562
    
    def crt(self, width):
        return width * .75

    def wide_door(self, width):
        return width * 1.25

    def narrow_door(self, width):
        return width * 1.5
    

fig_dims = FigDimensions()


def generate_graph_matplotlib(df: pd.DataFrame, show_key: str, num_clusters: int, matrix = None):
    # init figure
    plt.rcParams['figure.figsize'] = [14.0, 8.0]
    plt.rcParams['figure.autolayout'] = True
    fig = plt.figure()  # make sure to call this, in order to create a new figure

    # draw figure
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)

    # vis_dims2 = tsne.fit_transform(matrix)

    new_df = df.copy()
    new_df = new_df.drop(['doc_id', 'Cluster'], axis=1)
    new_df.columns = new_df.columns.astype(str)
    vis_dims2 = tsne.fit_transform(new_df)

    # print(f'len(vis_dims2)={len(vis_dims2)} vis_dims2={vis_dims2}')

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    # print(f'len(x)={len(x)} x={x}')
    # print(f'len(y)={len(y)} y={y}')

    # fp = f'clusters_{show_key}_{num_clusters}.csv'
    # df.to_csv(fp)

    for category, color in enumerate(colors[:num_clusters]):
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)

        # print(f'category={category}, color={color}')
        # print(f'xs={xs}')
        # print(f'ys={ys}')

        avg_x = xs.mean()
        avg_y = ys.mean()
        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)

    # for i, row in df.iterrows():
    i = 0
    for doc_id, _ in df['doc_id'].items():
        # print(f'i={i} doc_id={doc_id}, vis_dims2[i][0]={vis_dims2[i][0]} vis_dims2[i][0]={vis_dims2[i][1]}')
        plt.annotate(doc_id, (vis_dims2[i][0], vis_dims2[i][1]))
        i += 1

    plt.title(f'{num_clusters} clusters for {show_key} visualized in 2D using t-SNE')

    # store figure
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    return img_buf



def generate_graph_plotly(doc_clusters_df: pd.DataFrame, episodes_df: pd.DataFrame, show_key: str, num_clusters: int) -> go.Figure:
    fig_width = fig_dims.MD11
    fig_height = fig_dims.hdef(fig_width)
    base_fig_title = f'{num_clusters} clusters for {show_key} visualized in 2D using t-SNE'

    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    new_df = doc_clusters_df.copy()
    new_df = new_df.drop(['doc_id', 'Cluster'], axis=1)
    new_df.columns = new_df.columns.astype(str)
    vis_dims2 = tsne.fit_transform(new_df)

    doc_clusters_df['cluster_color'] = doc_clusters_df['Cluster'].apply(lambda x: colors[x])

    merged_df = pd.merge(doc_clusters_df, episodes_df, on='doc_id', how='outer')
    # doc_clusters_df.join(episodes_df, on='doc_id', how='left')

    # print(f'merged_df={merged_df}')

    # TODO need to verify that embeddings exist for all documents or remove those lacking embeddings, otherwise array lengths done line up

    custom_data = ['title', 'season', 'sequence_in_season', 'air_date', 'episode_key']
    hover_data = {'title': True, 'season': True, 'sequence_in_season': True, 'air_date': True, 'episode_key': True}

    # print(f'len(vis_dims2)={len(vis_dims2)} vis_dims2={vis_dims2}')

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    # print(f'len(x)={len(x)} x={x}')
    # print(f'len(y)={len(y)} y={y}')

    # init figure with core properties
    fig = px.scatter(merged_df, x=x, y=y, color=merged_df.cluster_color,
                title=base_fig_title, custom_data=custom_data,
                hover_name=merged_df.doc_id, hover_data=hover_data,
                height=fig_height, width=fig_width, opacity=0.7)
    # axis metadata
    fig.update_xaxes(title_text='x axis of t-SNE')
    fig.update_yaxes(title_text='y axis of t-SNE')

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