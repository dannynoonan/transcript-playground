import copy
import igraph as ig
import io
import math
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import random
from sklearn.manifold import TSNE

import app.es.es_read_router as esr
from app.show_metadata import ShowKey, show_metadata
from app.web.fig_helper import apply_animation_settings
import app.web.fig_metadata as fm


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


def build_cluster_scatter(episode_embeddings_clusters_df: pd.DataFrame, show_key: str, num_clusters: int) -> go.Figure:
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


def build_3d_network_graph(show_key: str, data: dict) -> go.Figure:
    print(f'in build_3d_network_graph show_key={show_key}')

    # reference: https://plotly.com/python/v3/3d-network-graph/
    
    # data = esr.episode_relations_graph(ShowKey(show_key), model_vendor, model_version, max_edges=max_edges)

    N=len(data['nodes'])

    L=len(data['links'])
    Edges=[(data['links'][k]['source'], data['links'][k]['target']) for k in range(L)]

    G=ig.Graph(Edges, directed=False)

    labels=[]
    group=[]
    for node in data['nodes']:
        labels.append(node['name'])
        group.append(node['group'])

    layt=G.layout('kk', dim=3)

    Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[layt[k][1] for k in range(N)]# y-coordinates
    Zn=[layt[k][2] for k in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in Edges:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]
        Ze+=[layt[e[0]][2],layt[e[1]][2], None]

    trace1=go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        line=dict(
            color='rgb(125,125,125)', 
            width=1
        ),
        hoverinfo='none'
    )

    trace2=go.Scatter3d(
        x=Xn,
        y=Yn,
        z=Zn,
        mode='markers',
        name='actors',
        marker=dict(
            symbol='circle',
            size=6,
            color=group,
            colorscale='Viridis',
            line=dict(
                color='rgb(50,50,50)', 
                width=0.5
            )
        ),
        text=labels,
        hoverinfo='text'
    )

    axis=dict(showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title=''
    )

    fig_width = fm.fig_dims.MD10
    fig_height = fm.fig_dims.hdef(fig_width)

    layout = go.Layout(
        title="placeholder title",
        width=fig_width,
        height=fig_height,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
        margin=dict(t=100),
        hovermode='closest',
        annotations=[
            dict(
                showarrow=False,
                text="placeholder text",
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=14)
            )
        ],    
    )
    data=[trace1, trace2]
    fig=go.Figure(data=data, layout=layout)

    return fig    


def build_episode_gantt(show_key: str, data: list) -> go.Figure:
    print(f'in build_episode_gantt show_key={show_key}')

    '''
    reference:
    https://plotly.com/python/gantt/
    https://stackoverflow.com/questions/73247210/how-to-plot-a-gantt-chart-using-timesteps-and-not-dates-using-plotly
    https://community.plotly.com/t/gantt-chart-issue-legend-and-hover/8011
    '''

    df = pd.DataFrame(data)

    span_keys = df.Task.unique()
    keys_to_colors = {}
    colors_to_keys = {}
    for sk in span_keys:
        r = random.randrange(255)
        g = random.randrange(255)
        b = random.randrange(255)
        rgb = f'rgb({r},{g},{b})'
        keys_to_colors[sk] = rgb
        colors_to_keys[rgb] = sk

    fig = ff.create_gantt(df, index_col='Task', bar_width=0.1, colors=keys_to_colors, group_tasks=True, 
                          title='Character dialog over duration of episode') # TODO change this for locations
    fig.update_layout(xaxis_type='linear', autosize=False)

    # inject dialog into hover 'text' property
    for gantt_row in fig['data']:
        if 'text' in gantt_row and gantt_row['text'] and len(gantt_row['text']) > 0:
            # once gantt figure is generated, speaker and location info is distributed across figure 'data' elements, and 'name' is not stored for every row.
            # the rgb data stored in 'legendgroup' seems to be the only way to reverse lookup which speaker or location is being referenced, hence the colors_to_keys map.
            rgb_val = gantt_row['legendgroup'].replace(' ', '')
            gantt_row_key = colors_to_keys[rgb_val]
            # the 'text' of a gantt row is stored in an unnamed 'data' element (associated to a gantt row via its 'legendgroup' rgb color) as a tuple, making it immutable.
            # rather than updating it index-by-index, it must be copied, cast as a list, mutated iteratively, and updated in one swoop via gantt_row.update at the end.
            gantt_row_text = list(gantt_row['text'])
            for i in range(len(gantt_row['x'])):
                word_i = gantt_row['x'][i]
                start_row = df.loc[(df['Task'] == gantt_row_key) & (df['Start'] == word_i)]
                if len(start_row) > 0 and 'Line' in start_row.iloc[0]:
                    gantt_row_text[i] = start_row.iloc[0]['Line']
                    continue
                finish_row = df.loc[(df['Task'] == gantt_row_key) & (df['Finish'] == word_i)]
                if len(finish_row) > 0 and 'Line' in finish_row.iloc[0]:
                    gantt_row_text[i] = finish_row.iloc[0]['Line']

            gantt_row.update(text=gantt_row_text, hoverinfo='all') # TODO hoverinfo='text+y' would remove word index
    
    return fig


def build_series_gantt(show_key: str, data: list, type: str) -> go.Figure:
    print(f'in build_show_gantt show_key={show_key} type={type}')

    # TODO where/how should this truncation happen?
    if type == 'speakers':
        title='Character continuity over duration of series'
        trimmed_data = []
        for d in data:
            if d['Task'] in show_metadata[show_key]['regular_cast']:
                trimmed_data.append(d)
        data = trimmed_data
    elif type == 'locations':
        title='Scene location continuity over course of series'
    elif type == 'topics':
        title='Topics over course of series'

    df = pd.DataFrame(data)

    if type == 'topics':
        df = df.sort_values('Task')

    span_keys = df.Task.unique()
    keys_to_colors = {}
    colors_to_keys = {}
    for sk in span_keys:
        r = random.randrange(255)
        g = random.randrange(255)
        b = random.randrange(255)
        rgb = f'rgb({r},{g},{b})'
        keys_to_colors[sk] = rgb
        colors_to_keys[rgb] = sk

    fig = ff.create_gantt(df, index_col='Task', bar_width=0.1, colors=keys_to_colors, group_tasks=True, title=title, height=1000) # TODO scale height to number of rows
    fig.update_layout(xaxis_type='linear', autosize=False)

    # inject dialog into hover 'text' property
    for gantt_row in fig['data']:
        if 'text' in gantt_row and gantt_row['text'] and len(gantt_row['text']) > 0:
            # once gantt figure is generated, speaker and location info is distributed across figure 'data' elements, and 'name' is not stored for every row.
            # the rgb data stored in 'legendgroup' seems to be the only way to reverse lookup which speaker or location is being referenced, hence the colors_to_keys map.
            rgb_val = gantt_row['legendgroup'].replace(' ', '')
            gantt_row_key = colors_to_keys[rgb_val]
            # the 'text' of a gantt row is stored in an unnamed 'data' element (associated to a gantt row via its 'legendgroup' rgb color) as a tuple, making it immutable.
            # rather than updating it index-by-index, it must be copied, cast as a list, mutated iteratively, and updated in one swoop via gantt_row.update at the end.
            gantt_row_text = list(gantt_row['text'])
            for i in range(len(gantt_row['x'])):
                episode_i = gantt_row['x'][i]
                start_row = df.loc[(df['Task'] == gantt_row_key) & (df['Start'] == episode_i)]
                if len(start_row) > 0 and 'info' in start_row.iloc[0]:
                    gantt_row_text[i] = start_row.iloc[0]['info']
                    continue
                # finish_row = df.loc[(df['Info'] == gantt_row_key) & (df['Finish'] == episode_i)]
                # if len(finish_row) > 0 and 'Info' in finish_row.iloc[0]:
                #     gantt_row_text[i] = finish_row.iloc[0]['Info']

            gantt_row.update(text=gantt_row_text, hoverinfo='all') # TODO hoverinfo='text+y' would remove word index
    
    return fig


def build_series_search_results_gantt(show_key: str, qt: str, matching_episodes: list, episode_speakers_sequence: list) -> go.Figure:
    print(f'in build_series_search_results_gantt show_key={show_key} qt={qt} len(matching_episodes)={len(matching_episodes)}, len(episode_speakers_sequence)={len(episode_speakers_sequence)}')

    # load full time-series sequence of speakers by episode into a dataframe
    df = pd.DataFrame(episode_speakers_sequence)
    df['matching_line_count'] = 0
    df['matching_lines'] = np.NaN
    speakers_to_keep = []
    # for each matching episode, concat lines and tally line_count per speaker, then insert into corresponding row in df 
    for episode in matching_episodes:
        speakers_to_lines = {}
        speakers_to_line_counts = {}
        for scene in episode['scenes']:
            for scene_event in scene['scene_events']:
                speaker = scene_event['spoken_by']
                if speaker not in speakers_to_lines:
                    speakers_to_lines[speaker] = []
                    speakers_to_line_counts[speaker] = 0
                    if speaker not in speakers_to_keep:
                        speakers_to_keep.append(speaker)
                speakers_to_lines[speaker].append(f"[S{scene['sequence']+1}] {scene_event['dialog']}\n\n")  # TODO newlines not working
                # speakers_to_lines[speaker].append(f"{scene_event['dialog']}\n\n")
                speakers_to_line_counts[speaker] += 1
        for speaker, _ in speakers_to_line_counts.items():
            df.loc[(df['Task'] == speaker) & (df['episode_key'] == episode['episode_key']), 'matching_line_count'] = speakers_to_line_counts[speaker]
            df.loc[(df['Task'] == speaker) & (df['episode_key'] == episode['episode_key']), 'matching_lines'] = ''.join(speakers_to_lines[speaker])
    
    speakers_to_keep = list(dict.fromkeys(speakers_to_keep))
    # only keep rows for speakers that have at least 1 match
    df = df.loc[df['Task'].isin(speakers_to_keep)]
    # if `matching_line_count` > 0:
    #   - mark `highlight` column yes/no: tells ff.create_gantt which color to use (gray or highlighted) via `index_col` 
    #   - set `hover_text` column with episode and matching_line data for hover display 
    df['highlight'] = df['matching_line_count'].apply(lambda x: 'yes' if x > 0 else 'no')
    matching_lines_df = df[df['highlight'] == 'yes']
    matching_lines_df['hover_text'] = matching_lines_df['episode_title'] + ':\n\n' + matching_lines_df['matching_lines']  # TODO newlines not working
    # (*) this feels a little fragile, but the sequence and index positions of the `hover_text` list map precisely 1:2 to the sequence and index positions 
    # of the gantt data rows in fig['data'] below, because each speaker-episode element maps to two gantt row entries (a Start entry and a Finish entry)
    hover_text = list(matching_lines_df['hover_text'])

    # file_path = f'./app/data/test_series_search_results_gantt_{show_key}.csv'
    # df.to_csv(file_path)

    fig = ff.create_gantt(df, index_col='highlight', bar_width=0.1, colors=['#B0B0B0', '#FF0000'], group_tasks=True, height=1000) # TODO scale height to number of rows
    fig.update_layout(xaxis_type='linear', autosize=False)

    # inject dialog stored in `hover_text` list into fig['data'] `text` property
    for gantt_row in fig['data']:
        print(gantt_row)
        if 'text' in gantt_row and gantt_row['text'] and len(gantt_row['text']) > 0 and gantt_row['legendgroup'] == 'rgb(255, 0, 0)':
            # once gantt figure is generated, speaker and location info is distributed across figure 'data' elements, and 'name' is not stored for every row.
            # the rgb data stored in 'legendgroup' seems to be the only way to reverse lookup which speaker or location is being referenced.
            # the 'text' of a gantt row is stored in an unnamed 'data' element (associated to a gantt row via its 'legendgroup' rgb color) as a tuple, making it immutable.
            # rather than updating it index-by-index, it must be copied, cast as a list, mutated iteratively, and updated in one swoop via gantt_row.update at the end.
            gantt_row_text = list(gantt_row['text'])
            for i in range(len(gantt_row['x'])):
                # (*) mentioned above: the sequence and index positions of `hover_text` list map 1:2 to sequence and index positions of gantt rows in fig['data']
                gantt_row_text[i] = hover_text[math.floor(i/2)]

            gantt_row.update(text=gantt_row_text, hoverinfo='all') # TODO hoverinfo='text+y' would remove episode index
    
    return fig


def build_speaker_line_chart(show_key: str, df: pd.DataFrame, span_granularity: str, aggregate_ratio: bool = False, season: str = None) -> go.Figure:
    print(f'in build_speaker_line_chart show_key={show_key} span_granularity={span_granularity} aggregate_ratio={aggregate_ratio} season={season}')

    y = f'{span_granularity}_count'
    if aggregate_ratio or span_granularity == 'episode':
        if season:
            y = f'{y}_pct_of_season'
        else:
            y = f'{y}_pct_of_series'

    if season:
        df = df.loc[df['season'] == season]

    # drop any speaker having no span data during the episode range
    speaker_span_aggs = df.groupby('speaker')[y].sum()
    for key, value in speaker_span_aggs.to_dict().items():
        if value == 0:
            df = df[df['speaker'] != key]

    custom_data = ['speaker', 'episode_title', 'season', 'sequence_in_season']

    fig = px.line(df, x='episode_i', y=y, color='speaker', height=800, render_mode='svg', line_shape='spline',
                  custom_data=custom_data)

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{customdata[0]}: %{y:.2f}</b>",
            "Season %{customdata[2]}, Episode %{customdata[3]}:",
            "\"%{customdata[1]}\"",
        ])
    )

    return fig


def build_location_line_chart(show_key: str, df: pd.DataFrame, span_granularity: str, aggregate_ratio: bool = False, season: str = None) -> go.Figure:
    print(f'in build_location_line_chart show_key={show_key} span_granularity={span_granularity} aggregate_ratio={aggregate_ratio} season={season}')

    y = f'{span_granularity}_count'
    if aggregate_ratio or span_granularity == 'episode':
        if season:
            y = f'{y}_pct_of_season'
        else:
            y = f'{y}_pct_of_series'

    if season:
        df = df.loc[df['season'] == season]

    # drop any location having no span data during the episode range
    location_span_aggs = df.groupby('location')[y].sum()
    for key, value in location_span_aggs.to_dict().items():
        if value == 0:
            df = df[df['location'] != key]

    custom_data = ['location', 'episode_title', 'season', 'sequence_in_season']

    fig = px.line(df, x='episode_i', y=y, color='location', height=800, render_mode='svg', line_shape='spline',
                  custom_data=custom_data)

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{customdata[0]}: %{y:.2f}</b>",
            "Season %{customdata[2]}, Episode %{customdata[3]}:",
            "\"%{customdata[1]}\"",
        ])
    )

    return fig


def build_speaker_frequency_bar(show_key: str, df: pd.DataFrame, span_granularity: str, aggregate_ratio: bool, season: int, 
                                sequence_in_season: int = None, animate: bool = False) -> go.Figure:
    print(f'in build_speaker_frequency_bar show_key={show_key} span_granularity={span_granularity} aggregate_ratio={aggregate_ratio} season={season} sequence_in_season={sequence_in_season} animate={animate}')

    animation_frame = None

    # in this context:
    #   - `aggregate_ratio=True`: intra-season episode-by-episode tabulation using sum()
    #   - `aggregate_ratio=False`: inter-season comparison between totals using max() (and `sequence_in_season` is ignored)
    if aggregate_ratio:
        if animate:
            animation_frame = 'sequence_in_season'
            df = df.loc[df['season'] == season]
            x = f'{span_granularity}_count_pct_of_season'
            sum_df = df
        else:
            if not season:
                season = 1
            if not sequence_in_season:
                sequence_in_season = 1
            df = df.loc[df['season'] == season]
            df = df.loc[df['sequence_in_season'] <= sequence_in_season]
            x = f'{span_granularity}_count'
            # if `span_granularity='episode'` dynamically populate `episode_count` column using `scene_count` column to enable episode tabulation
            if span_granularity == 'episode':
                df['episode_count'] = df['scene_count'].apply(lambda x: 1 if x > 0 else 0)
            sum_df = df.groupby(['speaker', 'season'], as_index=False)[x].sum()   
    else:
        if animate:
            animation_frame = 'season'
        else:
            if season:
                df = df.loc[df['season'] == season]
        x = f'{span_granularity}_count'
        # if `span_granularity='episode'` dynamically populate `episode_count` column using `scene_count` column to enable episode tabulation
        if span_granularity == 'episode':
            df['episode_count'] = df['scene_count'].apply(lambda x: 1 if x > 0 else 0)
        sum_df = df.groupby(['speaker', 'season'], as_index=False)[x].sum()

    # sum_df.sort_values(['season', x], ascending=[True, False], inplace=True)
    # category_orders = {'speaker': sum_df['speaker'].unique()}

    # if animate:
    #     file_path = f'./app/data/speaker_frequency_bar_{show_key}_{span_granularity}_animation.csv'
    # else:
    #     file_path = f'./app/data/speaker_frequency_bar_{show_key}_{span_granularity}_{season}_{sequence_in_season}.csv'
    # sum_df.to_csv(file_path)

    # custom_data = []  # TODO

    fig = px.bar(sum_df, x=x, y='speaker', color='speaker',
                # custom_data=custom_data, hover_name=cols.VOTE_WEIGHT, hover_data=hover_data,
                # text=cols.EC_VOTES, 
                 animation_frame=animation_frame, # ignored if df is for single year
                #  category_orders=category_orders,
                # color_discrete_map=color_discrete_map, category_orders=category_orders,
                # labels={cols.GROUP: groups_label},
                # range_x=[vw_min,vw_max], log_x=True, height=fig_height
    )

    fig.update_layout(showlegend=False)

    fig.update_layout(
        yaxis={'tickangle': 35, 'showticklabels': True, 'type': 'category', 'tickfont_size': 8},
        yaxis_categoryorder='total ascending') # yaxis_categoryorder
 
    if animate:
        apply_animation_settings(fig, 'TODO', frame_rate=1500)
    
    return fig


# def build_speaker_line_chart(show_key: str, data: list, aggregate_ratio: bool = False) -> go.Figure:
#     print(f'in build_speaker_line_chart show_key={show_key}')

#     df = pd.DataFrame(data)

#     y='Span'

#     if aggregate_ratio:
#         df['Span_Ratio'] = df['Span'] / df['Denominator']
#         y='Span_Ratio'

#     # drop any speaker having no span data during the episode range
#     speaker_span_aggs = df.groupby('Speaker')['Span'].sum()
#     for key, value in speaker_span_aggs.to_dict().items():
#         if value == 0:
#             df = df[df['Speaker'] != key]

#     # span_keys = df.Task.unique()
#     # keys_to_colors = {}
#     # colors_to_keys = {}
#     # for sk in span_keys:
#     #     r = random.randrange(255)
#     #     g = random.randrange(255)
#     #     b = random.randrange(255)
#     #     rgb = f'rgb({r},{g},{b})'
#     #     keys_to_colors[sk] = rgb
#     #     colors_to_keys[rgb] = sk

#     fig = px.line(df, x='Episode_i', y=y, color='Speaker', height=800, render_mode='svg', line_shape='spline')

#     return fig


def build_network_graph() -> go.Figure:
    print(f'in build_network_graph')

    # reference https://plotly.com/python/network-graphs/

    G = nx.random_geometric_graph(200, 0.125)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title='<br>Network graph made with Python',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="annotation text",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig
