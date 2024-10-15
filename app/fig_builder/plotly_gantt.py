import math
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import random

import app.fig_meta.color_meta as cm
import app.fig_meta.gantt_helper as gh
from app.show_metadata import show_metadata


def build_episode_gantt(show_key: str, y_axis: str, timeline_data: list, interval_data: list = None) -> go.Figure:
    print(f'in build_episode_gantt show_key={show_key} y_axis={y_axis}')

    '''
    reference:
    https://plotly.com/python/gantt/
    https://stackoverflow.com/questions/73247210/how-to-plot-a-gantt-chart-using-timesteps-and-not-dates-using-plotly
    https://community.plotly.com/t/gantt-chart-issue-legend-and-hover/8011
    '''

    df = pd.DataFrame(timeline_data)
    speaker_colors = cm.flatten_speaker_colors(show_key, to_rgb=True)
    # if y_axis == 'speakers':
    #     # title = 'Character dialog timeline'
    # elif y_axis == 'locations':
    #     # title = 'Scene location timeline'
    # else:
    #     # title = 'Placeholder title'

    x_label = 'line of dialog'

    # tighten x axis margins to 2% of the x axis span
    x_max = df.Finish.max()
    buffer = x_max * 0.02
    x_low = -buffer
    x_high = x_max + buffer

    span_keys = df.Task.unique()
    keys_to_colors = {}
    colors_to_keys = {}
    for sk in span_keys:
        if y_axis == 'speakers' and sk in speaker_colors:
            rgb = speaker_colors[sk]
        else:
            r = random.randrange(255)
            g = random.randrange(255)
            b = random.randrange(255)
            rgb = f'rgb({r},{g},{b})'
        keys_to_colors[sk] = rgb
        colors_to_keys[rgb] = sk

    fig = ff.create_gantt(df, index_col='Task', bar_width=0.1, colors=keys_to_colors, group_tasks=True, title=None)

    fig.update_layout(xaxis_type='linear', xaxis_title=x_label, autosize=False, margin=dict(r=30, t=30, b=60))
    fig.update_xaxes(range=[x_low, x_high])
    fig.update_yaxes(autorange=True)

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

    shapes = []
    # if interval_data, add markers and labels designating intervals 
    if interval_data:
        if y_axis == 'speakers':
            interval_shapes = gh.build_and_annotate_scene_labels(fig, interval_data)
            shapes.extend(interval_shapes)
            # scene_blocks = build_and_annotate_scene_blocks(interval_data)
            # shapes.extend(scene_blocks)
    fig.update_layout(shapes=shapes)
    
    return fig


def build_series_gantt(show_key: str, df: pd.DataFrame, y_axis: str, interval_data: dict = None) -> go.Figure:
    print(f'in build_series_gantt show_key={show_key} y_axis={y_axis}')

    speaker_colors = cm.flatten_speaker_colors(show_key, to_rgb=True)

    x_label = 'episode number'

    # tighten x axis margins to 2% of the x axis span
    x_max = df.Finish.max()
    buffer = x_max * 0.02
    x_low = -buffer
    x_high = x_max + buffer

    if y_axis == 'speakers':
        title = 'Character continuity over course of series'
        # limit speaker gantt to those in `speakers` index (for visual layout, and only slightly for page load performance)
        # matches = esr.fetch_indexed_speakers(ShowKey(show_key), min_episode_count=2)
        # speakers = [m['speaker'] for m in matches['speakers']]
        speakers = list(show_metadata[show_key]['regular_cast'].keys()) + list(show_metadata[show_key]['recurring_cast'].keys())
        df = df.loc[df['Task'].isin(speakers)]
    elif y_axis == 'locations':
        title = 'Scene location continuity over course of series'
    elif y_axis == 'topics':
        title = 'Episode genres over course of series'

    if y_axis == 'topics':
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
            rgb = cm.topic_cat_rank_color_mapper(topic_cats.index(cat), hex_hue)
            keys_to_colors[cat_rank] = rgb
            colors_to_keys[rgb] = cat_rank
        fig_height = 200 + len(df['Task'].unique()) * 20

    else: # ['speakers', 'locations']
        index_col = 'Task'
        span_keys = df.Task.unique()
        keys_to_colors = {}
        colors_to_keys = {}
        for sk in span_keys:
            if y_axis == 'speakers' and sk in speaker_colors:
                rgb = speaker_colors[sk]
            else:
                r = random.randrange(255)
                g = random.randrange(255)
                b = random.randrange(255)
                rgb = f'rgb({r},{g},{b})'
            keys_to_colors[sk] = rgb
            colors_to_keys[rgb] = sk
        fig_height = 200 + len(colors_to_keys) * 20
    
    fig = ff.create_gantt(df, index_col=index_col, bar_width=0.1, colors=keys_to_colors, group_tasks=True, title=title, height=fig_height) # TODO scale height to number of rows
    
    fig.update_layout(xaxis_type='linear', xaxis_title=x_label, autosize=False, margin=dict(r=30, t=60, b=60))
    fig.update_xaxes(range=[x_low, x_high])

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
                start_row = df.loc[(df[index_col] == gantt_row_key) & (df['Start'] == episode_i)]
                if len(start_row) > 0 and 'info' in start_row.iloc[0]:
                    gantt_row_text[i] = start_row.iloc[0]['info']
                    continue
                # if type == 'topics': 
                finish_row = df.loc[(df[index_col] == gantt_row_key) & (df['Finish'] == episode_i)]
                if len(finish_row) > 0 and 'info' in finish_row.iloc[0]:
                    gantt_row_text[i] = finish_row.iloc[0]['info']

            gantt_row.update(text=gantt_row_text, hoverinfo='all') # TODO hoverinfo='text+y' would remove word index

    shapes = []
    # if interval_data, add markers and labels designating intervals 
    if interval_data:
        interval_shapes = gh.build_and_annotate_season_labels(fig, interval_data)
        shapes.extend(interval_shapes)
    fig.update_layout(shapes=shapes)
    
    return fig


def build_episode_search_results_gantt(show_key: str, timeline_df: pd.DataFrame, matching_lines_df: pd.DataFrame) -> go.Figure:
    print(f'in build_episode_search_results_gantt show_key={show_key} len(timeline_df)={len(timeline_df)}, len(matching_lines_df)={len(matching_lines_df)}')

    x_label = 'line of dialog'

    # tighten x axis margins to 2% of the x axis span
    x_max = timeline_df.Finish.max()
    buffer = x_max * 0.02
    x_low = -buffer
    x_high = x_max + buffer
    
    # (*) this feels a little fragile, but the sequence and index positions of the `hover_text` list map precisely 1:2 to the sequence and index positions 
    # of the gantt data rows in fig['data'] below, because each speaker-episode element maps to two gantt row entries (a Start entry and a Finish entry)
    hover_text = list(matching_lines_df['Line'])

    # scale height to number of rows
    fig_height = 200 + len(timeline_df['Task'].unique()) * 20

    fig = ff.create_gantt(timeline_df, index_col='highlight', bar_width=0.1, colors=['#B0B0B0', '#FF0000'], group_tasks=True, height=fig_height, title=None)

    fig.update_layout(xaxis_type='linear', autosize=False, xaxis_title=x_label, margin=dict(r=30, t=30, b=60))
    fig.update_xaxes(range=[x_low, x_high])
    fig.update_yaxes(autorange=True)

    # inject dialog stored in `hover_text` list into fig['data'] `text` property
    for gantt_row in fig['data']:
        # print(gantt_row)
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


def build_series_search_results_gantt(show_key: str, timeline_df: pd.DataFrame, matching_episodes: list, interval_data: dict = None) -> go.Figure:
    print(f'in build_series_search_results_gantt show_key={show_key} len(timeline_df)={len(timeline_df)} len(matching_episodes)={len(matching_episodes)}')

    x_label = 'episode'

    # tighten x axis margins to 2% of the x axis span
    x_max = timeline_df.Finish.max()
    buffer = x_max * 0.02
    x_low = -buffer
    x_high = x_max + buffer

    timeline_df['matching_line_count'] = 0
    timeline_df['matching_lines'] = np.NaN
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
                speakers_to_lines[speaker].append(f"[scene {scene['sequence']+1}] {scene_event['dialog']}") 
                # speakers_to_lines[speaker].append(f"{scene_event['dialog']}\n\n")
                speakers_to_line_counts[speaker] += 1
        for speaker, _ in speakers_to_line_counts.items():
            timeline_df.loc[(timeline_df['Task'] == speaker) & (timeline_df['episode_key'] == episode['episode_key']), 'matching_line_count'] = speakers_to_line_counts[speaker]
            timeline_df.loc[(timeline_df['Task'] == speaker) & (timeline_df['episode_key'] == episode['episode_key']), 'matching_lines'] = '<br>'.join(speakers_to_lines[speaker])
            # timeline_df.loc[(timeline_df['Task'] == speaker) & (timeline_df['episode_key'] == episode['episode_key']), 'score'] = episode['score']
            timeline_df.loc[(timeline_df['Task'] == speaker) & (timeline_df['episode_key'] == episode['episode_key']), 'agg_score'] = episode['agg_score']
    
    speakers_to_keep = list(dict.fromkeys(speakers_to_keep)) 
    # only keep rows for speakers that have at least 1 match
    timeline_df = timeline_df.loc[timeline_df['Task'].isin(speakers_to_keep)]
    # if `matching_line_count` > 0:
    #   - mark `highlight` column yes/no: tells ff.create_gantt which color to use (gray or highlighted) via `index_col` 
    #   - set `hover_text` column with episode and matching_line data for hover display 
    timeline_df['highlight'] = timeline_df['matching_line_count'].apply(lambda x: 'yes' if x > 0 else 'no')
    matching_lines_df = timeline_df[timeline_df['highlight'] == 'yes']
    # concatenate fields into single 'hover_text' cell, in lieu of dynamically generated ff.crate_gantt hovertemplate
    matching_lines_df['hover_text'] = matching_lines_df.apply(lambda x: gh.search_results_hover_text(x), axis=1) 
    # (*) this feels a little fragile, but the sequence and index positions of the `hover_text` list map precisely 1:2 to the sequence and index positions 
    # of the gantt data rows in fig['data'] below, because each speaker-episode element maps to two gantt row entries (a Start entry and a Finish entry)
    hover_text = list(matching_lines_df['hover_text'])

    # file_path = f'./app/data/test_series_search_results_gantt_{show_key}.csv'
    # df.to_csv(file_path)

    # scale height to number of rows
    fig_height = 200 + len(timeline_df['Task'].unique()) * 18

    fig = ff.create_gantt(timeline_df, index_col='highlight', bar_width=0.1, colors=['#B0B0B0', '#FF0000'], group_tasks=True, height=fig_height, title=None)

    fig.update_layout(xaxis_type='linear', autosize=False, xaxis_title=x_label, margin=dict(r=30, t=30, b=60))
    fig.update_xaxes(range=[x_low, x_high])
    fig.update_yaxes(autorange=True)

    # inject dialog stored in `hover_text` list into fig['data'] `text` property
    for gantt_row in fig['data']:
        # print(gantt_row)
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

    shapes = []
    # if interval_data, add markers and labels designating intervals 
    if interval_data:
        interval_shapes = gh.build_and_annotate_season_labels(fig, interval_data)
        shapes.extend(interval_shapes)
    fig.update_layout(shapes=shapes)
    
    return fig
