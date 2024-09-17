from bertopic import BERTopic
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
from app.web.fig_helper import apply_animation_settings, topic_cat_rank_color_mapper, to_mbti_x, to_mbti_y, to_dnda_x, to_dnda_y
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


def build_series_gantt(show_key: str, df: pd.DataFrame, type: str) -> go.Figure:
    print(f'in build_show_gantt show_key={show_key} type={type}')

    if type == 'speakers':
        title='Character continuity over duration of series'
        # limit speaker gantt to those in `speakers` index (for visual layout, and only slightly for page load performance)
        # matches = esr.fetch_indexed_speakers(ShowKey(show_key), min_episode_count=2)
        # speakers = [m['speaker'] for m in matches['speakers']]
        speakers = show_metadata[show_key]['regular_cast'] + show_metadata[show_key]['recurring_cast']
        df = df.loc[df['Task'].isin(speakers)]
    elif type == 'locations':
        title='Scene location continuity over course of series'
    elif type == 'topics':
        title='Topics over course of series'

    if type == 'topics':
        df = df.sort_values(['Task', 'Start'])
        # file_path = f'build_series_gantt_{type}_{show_key}.csv'
        # df.to_csv(file_path)
        index_col = 'cat_rank'
        df['cat_rank'] = df['topic_cat'] + '_' + df['rank'].astype(str)
        topic_cats = list(df['topic_cat'].unique())
        ranks = df['rank'].unique()
        cat_ranks = df['cat_rank'].unique()
        keys_to_colors = {}
        colors_to_keys = {}
        for cat_rank in cat_ranks:
            cat = cat_rank.split('_')[0]
            rank = cat_rank.split('_')[1]
            hex_hue = round(255/len(ranks)) * int(rank)
            rgb = topic_cat_rank_color_mapper(topic_cats.index(cat), hex_hue)
            keys_to_colors[cat_rank] = rgb
            colors_to_keys[rgb] = cat_rank
        fig_height = 250 + len(df['Task'].unique()) * 25

    else: # ['speakers', 'locations']
        index_col = 'Task'
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
        fig_height = 250 + len(colors_to_keys) * 25
    
    fig = ff.create_gantt(df, index_col=index_col, bar_width=0.2, colors=keys_to_colors, group_tasks=True, title=title, height=fig_height) # TODO scale height to number of rows
    fig.update_layout(xaxis_type='linear', autosize=False)

    gantt_row_with_text_count = 0
    for gantt_row in fig['data']:
        if 'text' in gantt_row and gantt_row['text'] and len(gantt_row['text']) > 0:
            gantt_row_with_text_count += 1
            # once gantt figure is generated, speaker and location info is distributed across figure 'data' elements, and 'name' is not stored for every row.
            # the rgb data stored in 'legendgroup' seems to be the only way to reverse lookup which speaker or location is being referenced, hence the colors_to_keys map.
            rgb_val = gantt_row['legendgroup'].replace(' ', '')
            gantt_row_key = colors_to_keys[rgb_val]
            # the 'text' of a gantt row is stored in an unnamed 'data' element (associated to a gantt row via its 'legendgroup' rgb color) as a tuple, making it immutable.
            # rather than updating it index-by-index, it must be copied, cast as a list, mutated iteratively, and updated in one swoop via gantt_row.update at the end.
            gantt_row_text = list(gantt_row['text'])
            for i in range(len(gantt_row['x'])):
                episode_i = gantt_row['x'][i]
                start_row = df.loc[(df[index_col] == gantt_row_key) & (df['Start'] == episode_i)]
                if len(start_row) > 0 and 'info' in start_row.iloc[0]:
                    gantt_row_text[i] = start_row.iloc[0]['info']
                    continue
                # if type == 'topics': 
                finish_row = df.loc[(df[index_col] == gantt_row_key) & (df['Finish'] == episode_i)]
                if len(finish_row) > 0 and 'info' in finish_row.iloc[0]:
                    gantt_row_text[i] = finish_row.iloc[0]['info']

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

    file_path = f'./app/data/test_series_search_results_gantt_{show_key}.csv'
    df.to_csv(file_path)

    fig_height = 250 + len(df['Task'].unique()) * 25

    fig = ff.create_gantt(df, index_col='highlight', bar_width=0.1, colors=['#B0B0B0', '#FF0000'], group_tasks=True, height=fig_height) # TODO scale height to number of rows
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


def build_speaker_episode_frequency_bar(show_key: str, episode_key: str, df: pd.DataFrame, span_granularity: str) -> go.Figure:
    print(f'in build_speaker_frequency_bar show_key={show_key} episode_key={episode_key} span_granularity={span_granularity}')

    x = f'{span_granularity}_count'

    # custom_data = []  # TODO

    fig = px.bar(df, x=x, y='speaker', color='speaker',
                # custom_data=custom_data, hover_name=cols.VOTE_WEIGHT, hover_data=hover_data,
                #  category_orders=category_orders,
                # color_discrete_map=color_discrete_map, category_orders=category_orders,
                # labels={cols.GROUP: groups_label},
                # range_x=[vw_min,vw_max], log_x=True, height=fig_height
    )

    fig.update_layout(showlegend=False)

    fig.update_layout(
        yaxis={'tickangle': 35, 'showticklabels': True, 'type': 'category', 'tickfont_size': 8},
        yaxis_categoryorder='total ascending') # yaxis_categoryorder
    
    return fig


def build_episode_speaker_topic_scatter(show_key: str, episode_key: str, df: pd.DataFrame, topic_type: str) -> go.Figure:
    print(f'in build_episode_speaker_topic_scatter show_key={show_key} episode_key={episode_key} topic_type={topic_type}')

    ep_topic_key = f'ep_{topic_type}_topic_key'
    ep_topic_score = f'ep_{topic_type}_score'
    # ser_topic_key = f'ser_{topic_type}_topic_key'
    # ser_topic_score = f'ser_{topic_type}_score'

    shapes = []
    bgs = []

    if topic_type == 'mbti':
        topic_types = fm.mbti_types
        df['ep_x'] = df[ep_topic_key].apply(to_mbti_x)
        df['ep_y'] = df[ep_topic_key].apply(to_mbti_y)
        colors = ['orange', 'yellowgreen', 'crimson', 'mediumaquamarine']
        bgs = [[0, 2, 0, 2], [0, 2, 2, 4], [2, 4, 0, 2], [2, 4, 2, 4]]
        high_x = high_y = 4
    elif topic_type == 'dnda':
        topic_types = fm.dnda_types
        df['ep_x'] = df[ep_topic_key].apply(to_dnda_x)
        df['ep_y'] = df[ep_topic_key].apply(to_dnda_y)
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'yellowgreen', 'crimson']
        bgs = [[0, 1, 0, 1], [1, 2, 0, 1], [2, 3, 0, 1],
               [0, 1, 1, 2], [1, 2, 1, 2], [2, 3, 1, 2],
               [0, 1, 2, 3], [1, 2, 2, 3], [2, 3, 2, 3]]
        high_x = high_y = 3

    topics_to_counts = df[ep_topic_key].value_counts()

    topics_to_i = {t[0]:0 for t in topics_to_counts.items()}

    for index, row in df.iterrows():
        topic_key = row[ep_topic_key]
        topic_count = topics_to_counts[topic_key]
        topic_i = topics_to_i[topic_key]
        df.at[index, 'ep_x'] = row['ep_x'] + fm.topic_grid_coord_deltas[topic_count][topic_i][0]
        df.at[index, 'ep_y'] = row['ep_y'] + fm.topic_grid_coord_deltas[topic_count][topic_i][1]
        topics_to_i[topic_key] += 1
        
    fig = px.scatter(df, x='ep_x', y='ep_y', size=ep_topic_score, text='speaker',
                     range_x=[0,high_x], range_y=[0,high_y], width=800, height=600)
    
    for i, b in enumerate(bgs):
        shapes.append(dict(type="rect", x0=b[0], x1=b[1], y0=b[2], y1=b[3], fillcolor=colors[i], opacity=0.5, layer="below", line_width=0))
    fig.update_layout(shapes=shapes)

    fig.update_xaxes(showgrid=True, gridwidth=2, dtick="M2")
    fig.update_yaxes(showgrid=True, gridwidth=2, dtick="M2")

    fig.update_traces(textposition='top center')

    if topic_type == 'mbti':
        topic_types = fm.mbti_types
    elif topic_type == 'dnda':
        topic_types = fm.dnda_types

    for label, coords in topic_types.items():
        fig.add_annotation(text=label, x=(coords['coords'][0] + 0.2), y=(coords['coords'][1] - 0.1), 
                        showarrow=False, font=dict(family="arial", size=14, color="white"))
        fig.add_annotation(text=coords['descr'], x=(coords['coords'][0] + 0.5), y=(coords['coords'][1] - 0.9),
                        showarrow=False, font=dict(family="arial", size=14, color="white"))

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


def build_bertopic_model_3d_scatter(show_key: str, bertopic_model_id: str, bertopic_docs_df: pd.DataFrame) -> go.Figure:
    print(f'in build_bertopic_model_3d_scatter show_key={show_key} bertopic_model_id={bertopic_model_id}')

    custom_data = ['cluster_title_short', 'cluster', 'season', 'episode', 'title', 'speaker_group', 'topics_focused_tfidf_list']

    bertopic_docs_df['cluster_title_short_legend'] = bertopic_docs_df['cluster'].astype(str) + ' ' + bertopic_docs_df['cluster_title_short']
    # bertopic_docs_df['cluster_title_short_legend'] = bertopic_docs_df[['cluster', 'cluster_title_short']].apply(lambda x: ' '.join(str(x)), axis=1)

    fig = px.scatter_3d(bertopic_docs_df, x='x_coord', y='y_coord', z='z_coord', color='cluster_title_short_legend', opacity=0.7, custom_data=custom_data,
                        # labels={'Topic', 'Topic'}, color_discrete_map=color_discrete_map, category_orders=category_orders,
                        height=1000, width=1600)

    fig.update_traces(marker=dict(line=dict(width=0.1, color='DarkSlateGrey')), selector=dict(mode='markers'))

    fig.update_traces(
        hovertemplate = "".join([
            "<b>%{customdata[0]} (Topic %{customdata[1]})</b><br><br>",
            "<b>S%{customdata[2]}:E%{customdata[3]}: %{customdata[4]}</b><br>",            
            "Speaker group: %{customdata[5]}<br>",
            "Focal topics: %{customdata[6]}",
            "<extra></extra>"
        ]),
        # mode='markers',
        # marker={'sizemode':'area',
        #         'sizeref':10},
    )

    return fig


def build_bertopic_visualize_barchart(bertopic_model: BERTopic) -> go.Figure:
    '''
    Generate topic keyword barcharts using saved model file
    '''
    # fig = bertopic_model.visualize_barchart(top_n_topics=16, width=200, height=250)
    fig = bertopic_model.visualize_barchart(top_n_topics=16, width=400, height=300)

    # TODO saving
    # https://maartengr.github.io/BERTopic/api/plotting/barchart.html#bertopic.plotting._barchart.visualize_barchart
    # fig = topic_model.visualize_barchart()
    # fig.write_html("path/to/file.html")

    return fig


def build_bertopic_visualize_topics(bertopic_model: BERTopic) -> go.Figure:
    '''
    Generate topic graphs using saved model file
    '''
    fig = bertopic_model.visualize_topics(width=800, height=800)

    return fig


def build_bertopic_visualize_hierarchy(bertopic_model: BERTopic) -> go.Figure:
    '''
    Generate topic hierarchy using saved model file
    '''
    fig = bertopic_model.visualize_hierarchy(width=1600, height=1200)

    return fig


def build_episode_sentiment_line_chart(show_key: str, df: pd.DataFrame, speakers: list, emotions: list, focal_property: str) -> go.Figure:
    print(f'in build_sentiment_line_chart show_key={show_key} emotion={emotions} speakers={speakers} focal_property={focal_property}')

    # remove episode-level rows 
    df = df.loc[df['scene'] != 'ALL']
    # remove live-level rows 
    df = df.loc[df['line'] == 'ALL']
    # filter out all other emotions 
    df = df.loc[df['emotion'].isin(emotions)]
    # filter out all other speakers 
    df = df.loc[df['speaker'].isin(speakers)]
    # cast numeric columns correctly TODO should this happen upstream?
    df['scene'] = pd.to_numeric(df['scene'])
    df['score'] = pd.to_numeric(df['score'])

    # df = df.sort_values(['scene', 'speaker', 'emotion'])
    # print(df)

    fig = px.line(df, x='scene', y='score', color=focal_property, height=800, render_mode='svg', line_shape='spline')

    for i in range(len(fig.data)):
        fig.data[i].update(mode='markers+lines')

    return fig
