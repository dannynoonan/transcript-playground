import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE

import app.fig_builder.fig_helper as fh
import app.fig_builder.fig_metadata as fm


def build_episode_similarity_scatter(df: pd.DataFrame, seasons: list) -> go.Figure:
    print(f"in build_episode_similarity_scatter len(df)={len(df)} seasons={seasons}")
    
    # rename 'sequence_in_season' to 'episode' for display
    df.rename(columns={'sequence_in_season': 'episode'}, inplace=True)

    symbol_map = {'all': 'square', 'mlt': 'circle', 'focal': 'star'}

    # ad-hoc method of flattening topic metadata for hovertemplate display
    df['flattened_topics'] = df['topics_universal_tfidf'].apply(fh.flatten_topics)

    custom_data = ['title', 'season', 'episode', 'score', 'rank', 'focal_speakers', 'flattened_topics']
    
    fig = px.scatter(df, x='episode', y='season', size='rev_rank', color='score', symbol='group',
                     custom_data=custom_data, color_continuous_scale='viridis', symbol_map=symbol_map)
                    #  width=1000, height=500
                    
    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>S%{customdata[1]}, E%{customdata[2]}: \"%{customdata[0]}\"</b>",
            "Similarity score: %{customdata[3]:.2f} (#%{customdata[4]})",
            "Focal characters: %{customdata[5]}",
            "Categories: %{customdata[6]}"
        ])
    )    

    fig.update_yaxes(autorange="reversed")
    
    fig.update_layout(showlegend=False, margin=dict(t=30, b=60), 
                      yaxis=dict(tickmode='array', tickvals=seasons, ticktext=[f'Season {s}' for s in seasons]))

    fig['data'][1]['marker']['color'] = 'Silver'
    fig['data'][2]['marker']['color'] = 'Black'
    fig['data'][2]['marker']['line'] = dict(width=3, color='Yellow')

    return fig


def build_all_series_episodes_scatter(df: pd.DataFrame, seasons: list) -> go.Figure:
    print(f"in build_all_series_episodes_scatter len(df)={len(df)} seasons={seasons}")
    
    # rename 'sequence_in_season' to 'episode' for display
    df.rename(columns={'sequence_in_season': 'episode'}, inplace=True)

    # ad-hoc method of flattening topic metadata for hovertemplate display
    df['flattened_topics'] = df['topics_universal_tfidf'].apply(fh.flatten_topics)

    custom_data = ['title', 'season', 'episode', 'air_date', 'focal_speakers', 'flattened_topics']
    # custom_data = ['title', 'season', 'episode', 'focal_speakers', 'flattened_topics']
    
    fig = px.scatter(df, x='episode', y='season', custom_data=custom_data)
                    
    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>S%{customdata[1]}, E%{customdata[2]}: \"%{customdata[0]}\"</b>",
            "Date: %{customdata[3]}",
            "Focal characters: %{customdata[4]}",
            "Categories: %{customdata[5]}"
        ])
    )    

    fig.update_yaxes(autorange="reversed")
    
    fig.update_layout(showlegend=False, margin=dict(t=30, b=60), 
                      yaxis=dict(tickmode='array', tickvals=seasons, ticktext=[f'Season {s}' for s in seasons]))

    return fig


def build_speaker_topic_scatter(show_key: str, df: pd.DataFrame, topic_type: str, speaker_color_map: dict = None) -> go.Figure:
    print(f'in build_speaker_topic_scatter show_key={show_key} topic_type={topic_type}')

    if not speaker_color_map:
        speakers = df['speaker'].unique()
        speaker_color_map = fh.generate_speaker_color_discrete_map(show_key, speakers)

    shapes = []

    if topic_type == 'mbti':
        topic_types = fm.mbti_types
        df['ep_x'] = df['topic_key'].apply(fh.to_mbti_x)
        df['ep_y'] = df['topic_key'].apply(fh.to_mbti_y)
        title = "Myers-Briggs Temperaments"
        labels = {"ep_x": "←  personal                                            logical  →",
                  "ep_y": "←  present                                            possible  →"}
        high_x = high_y = 4
    elif topic_type == 'dnda':
        topic_types = fm.dnda_types
        df['ep_x'] = df['topic_key'].apply(fh.to_dnda_x)
        df['ep_y'] = df['topic_key'].apply(fh.to_dnda_y)
        title = "D & D Alignments"
        labels = {"ep_x": "←  evil                                                    good  →",
                  "ep_y": "←  chaotic                                               lawful  →"}
        high_x = high_y = 3

    topics_to_counts = df['topic_key'].value_counts()

    topics_to_i = {t[0]:0 for t in topics_to_counts.items()}

    df.sort_values('rank', inplace=True)

    for index, row in df.iterrows():
        topic_key = row['topic_key']
        topic_count = topics_to_counts[topic_key]
        # NOTE Ick. When there's overflow of speakers mapped to a particular topic, things get messy
        # The topic_count and topic_i counters are off by 1, so the checks here are gross
        # Ultimately we df.drop any overflow speaker (current cut-off is 9) mapped to the same topic
        if topic_count >= len(fm.topic_grid_coord_deltas):
            topic_count = len(fm.topic_grid_coord_deltas) - 1
        topic_i = topics_to_i[topic_key]
        # print(f'topic_key={topic_key} topics_to_counts[topic_key]={topics_to_counts[topic_key]} topic_count={topic_count} topic_i={topic_i}')
        if topic_i >= topic_count:
            # print(f"too many speakers mapped to topic_key={topic_key}, dropping row['speaker']={row['speaker']} from display")
            df.drop(index, inplace=True)
            continue
        df.at[index, 'ep_x'] = row['ep_x'] + fm.topic_grid_coord_deltas[topic_count][topic_i][0]
        df.at[index, 'ep_y'] = row['ep_y'] + fm.topic_grid_coord_deltas[topic_count][topic_i][1]
        topics_to_i[topic_key] += 1
        
    custom_data = ['speaker', 'rank', 'topic_key', 'score']

    fig = px.scatter(df, x='ep_x', y='ep_y', size='dot_size', size_max=15, color='speaker', text='rank', 
                     title=title, labels=labels, color_discrete_map=speaker_color_map, custom_data=custom_data,
                     range_x=[0,high_x], range_y=[0,high_y], height=800)

    # fig['layout']['yaxis']['scaleanchor'] = "x"
    
    for index, row in df.iterrows():
        fig.add_annotation(text=row['speaker'], x=row['ep_x'], y=(row['ep_y'] + 0.12),
                           showarrow=False, font=dict(family="Arial", size=9, color="Black"))
    
    for label, d in topic_types.items():
        # fig.add_annotation(text=label, x=(d['coords'][0] + 0.2), y=(d['coords'][3] - 0.1), 
        #                 showarrow=False, font=dict(family="arial", size=14, color="White"))
        fig.add_annotation(text=d['descr'], x=(d['coords'][0] + 0.5), y=(d['coords'][3] - 0.9),
                           showarrow=False, font=dict(family="Arial", size=14, color="White"))
        shapes.append(dict(type="rect", x0=d['coords'][0], x1=d['coords'][1], y0=d['coords'][2], y1=d['coords'][3],
                           fillcolor=d['color'], opacity=0.5, layer="below", line_width=0))
        
    # add speaker names above scatter points
    fig.update_layout(shapes=shapes, showlegend=True, legend_title=None, margin=dict(l=60, t=60, r=40, b=50),
                      title_font=dict(size=20), title_x=0.5, font=dict(family="Arial", size=9, color="Black"))
        
    if topic_type == 'mbti':
        fig.add_annotation(text='SF', x=0, y=0.15, showarrow=False, font=dict(family="Arial", size=18, color="Black"))
        fig.add_annotation(text='Relating', x=0, y=0, showarrow=False, font=dict(family="Arial", size=14, color="Black"))
        fig.add_annotation(text='NF', x=0, y=3.85, showarrow=False, font=dict(family="Arial", size=18, color="Black"))
        fig.add_annotation(text='Valuing', x=0, y=4, showarrow=False, font=dict(family="Arial", size=14, color="Black"))
        fig.add_annotation(text='ST', x=4, y=0.15, showarrow=False, font=dict(family="Arial", size=18, color="Black"))
        fig.add_annotation(text='Directing', x=4, y=0, showarrow=False, font=dict(family="Arial", size=14, color="Black"))
        fig.add_annotation(text='NT', x=4, y=3.85, showarrow=False, font=dict(family="Arial", size=18, color="Black"))
        fig.add_annotation(text='Visioning', x=4, y=4, showarrow=False, font=dict(family="Arial", size=14, color="Black"))
    
    # NOTE workaround to plotly bug where showticklabels=False (or tickvals=[]) forces x axis to top of figure
    # solution is to set showticklabels=True and set color to white
    # https://stackoverflow.com/questions/75350817/plotly-remove-axis-ticks-and-numbers-but-keep-label
    fig.update_xaxes(showgrid=True, gridwidth=2, dtick="M2", 
                     showticklabels=True, color='white', title_font={"size": 18, "color": "black"})
    fig.update_yaxes(showgrid=True, gridwidth=2, dtick="M2", 
                     showticklabels=True, color='white', title_font={"size": 18, "color": "black"})

    fig.update_traces(
        # textposition='top center',
        hovertemplate="<br>".join([
            "<b>%{customdata[0]}</b> (#%{customdata[1]})",
            "%{customdata[2]} score: %{customdata[3]:.2f}",
            "<extra></extra>"
        ])
    )    

    return fig


def build_speaker_chatter_scatter(df: pd.DataFrame, x_axis: str, y_axis: str) -> go.Figure:
    print(f'in build_speaker_chatter_scatter x_axis=`{x_axis}` y_axis=`{y_axis}`')

    fig = px.scatter(df, x=x_axis, y=y_axis, text='speaker', width=800, height=650)
                    #  trendline="ols", trendline_scope="overall", trendline_color_override="black")

    fig.update_traces(textposition='middle right')

    return fig


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
