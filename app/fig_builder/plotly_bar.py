import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app.fig_meta.layout_meta as lm
import app.fig_meta.color_meta as cm


def build_speaker_frequency_bar(show_key: str, df: pd.DataFrame, span_granularity: str, aggregate_ratio: bool, season: int, 
                                sequence_in_season: int = None, animate: bool = False) -> go.Figure:
    print(f'in build_speaker_frequency_bar show_key={show_key} span_granularity={span_granularity} aggregate_ratio={aggregate_ratio} season={season} sequence_in_season={sequence_in_season} animate={animate}')

    animation_frame = None
    title = 'Character chatter'

    speakers = df['speaker'].unique()
    color_discrete_map = cm.generate_speaker_color_discrete_map(show_key, speakers)

    # TODO 10/10/24 aggregate_ratio param and descriptions of usage are super confusion. Was this mainly designed for animation?
    # in this context:
    #   - `aggregate_ratio=True`: intra-season episode-by-episode tabulation using sum()
    #   - `aggregate_ratio=False`: inter-season comparison between totals using max() 
    # NOTE: previously, when `aggregate_ratio=False` the `sequence_in_season` param was ignored. Made sense for animation, but not episode-by-episode browsing
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
            episode_title = list(df['episode_title'].unique())[0] # gulp
            title = f'{title} in Season {season} Episode {sequence_in_season}: "{episode_title}"'
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
                title = f'{title} in Season {season}'
                if sequence_in_season:
                    df = df.loc[df['sequence_in_season'] == sequence_in_season]
                    episode_title = list(df['episode_title'].unique())[0] # gulp
                    title = f'{title} Episode {sequence_in_season}: "{episode_title}"'
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

    sum_df.rename(columns={'speaker': 'character'}, inplace=True)
    labels = {x: f'{span_granularity}s'}
    # custom_data = []  # TODO, though there isn't much value-add, just formatting and weird conditional logic

    fig = px.bar(sum_df, x=x, y='character', color='character', title=title, color_discrete_map=color_discrete_map, labels=labels, height=650,
                 animation_frame=animation_frame) # ignored if df is for single year
                # custom_data=custom_data, hover_data=hover_data, category_orders=category_orders,
                # range_x=[vw_min,vw_max], height=fig_height

    fig.update_layout(showlegend=False)

    fig.update_layout(
        yaxis={'tickangle': 35, 'showticklabels': True, 'type': 'category', 'tickfont_size': 8},
        yaxis_categoryorder='total ascending') # yaxis_categoryorder
 
    if animate:
        lm.apply_animation_settings(fig, 'TODO', frame_rate=1500)
    
    return fig


def build_speaker_episode_frequency_bar(show_key: str, df: pd.DataFrame, scale_by: str) -> go.Figure:
    print(f'in build_speaker_frequency_bar scale_by={scale_by}')

    speakers = df['character'].unique()
    color_discrete_map = cm.generate_speaker_color_discrete_map(show_key, speakers)

    custom_data = ['scenes', 'lines', 'words']

    fig = px.bar(df, x=scale_by, y='character', color='character', 
                 custom_data=custom_data, color_discrete_map=color_discrete_map)

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{y}</b>",
            "Scenes: %{customdata[0]}",
            "Lines: %{customdata[1]}",
            "Words: %{customdata[2]}",
            "<extra></extra>"
        ])
    )

    fig.update_layout(
        showlegend=False, margin=dict(l=60, t=30, r=30, b=60),
        yaxis={'tickangle': 35, 'showticklabels': True, 'type': 'category', 'tickfont_size': 8},
        yaxis_categoryorder='total ascending') # yaxis_categoryorder
    
    return fig


def build_speaker_series_hist(show_key: str, speaker: str, df: pd.DataFrame, granularity: str, color_col: str = None) -> go.Figure:
    print(f'in build_speaker_series_hist show_key={show_key} speaker={speaker} len(df)={len(df)} granularity={granularity} color_col={color_col}')

    if not color_col:
        color_col = 'season'

    custom_data = ['season', 'sequence_in_season', 'title', 'air_date', 'scene_count', 'line_count', 'word_count', 
                   'scene_count_rank', 'line_count_rank', 'word_count_rank', 'mbti', 'dnda', 'locations', 'companions']

    fig = px.bar(df, x='sequence', y=granularity, color=color_col, custom_data=custom_data,
                 range_x=[df['sequence'].min(), df['sequence'].max()])

    y_title = granularity.replace('_', ' ')

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>S%{customdata[0]}, E%{customdata[1]}: \"%{customdata[2]}\"</b> (%{customdata[3]})",
            "Scenes: %{customdata[4]} (#%{customdata[7]})",
            "Lines: %{customdata[5]} (#%{customdata[8]})",
            "Words: %{customdata[6]} (#%{customdata[9]})",
            "MBTI: %{customdata[10]}",
            "D&D: %{customdata[11]}",
            "Locations: %{customdata[12]}",
            "Companions: %{customdata[13]}",
            "<extra></extra>"
        ])
    )

    fig.update_layout(
        title=dict(text="episode", font=dict(size=14), y=0.1, x=0.5),
        xaxis=dict(showticklabels=False, visible=False),
        yaxis_title=y_title
    )
    
    return fig


def build_speaker_series_sentiment_hist(show_key: str, speaker: str, df: pd.DataFrame, emotion: str) -> go.Figure:
    print(f'in build_speaker_series_sentiment_hist show_key={show_key} speaker={speaker} len(df)={len(df)} emotion={emotion}')

    color_col = 'season'

    custom_data = ['season', 'sequence_in_season', 'title', 'air_date', 'scene_count', 'line_count', 'word_count', 
                   'scene_count_rank', 'line_count_rank', 'word_count_rank', 'mbti', 'dnda', 'locations', 'companions']
    
    # df['focus'] = emotion
    custom_data.extend(['emotion', 'score'])

    fig = px.bar(df, x='sequence', y='score', color=color_col, custom_data=custom_data)
                #  range_x=[df['sequence'].min(), df['sequence'].max()])

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{customdata[14]}: %{customdata[15]}</b><br>",
            "<b>S%{customdata[0]}, E%{customdata[1]}: \"%{customdata[2]}\"</b> (%{customdata[3]})",
            "Scenes: %{customdata[4]} (#%{customdata[7]})",
            "Lines: %{customdata[5]} (#%{customdata[8]})",
            "Words: %{customdata[6]} (#%{customdata[9]})",
            "MBTI: %{customdata[10]}",
            "D&D: %{customdata[11]}",
            "Locations: %{customdata[12]}",
            "Companions: %{customdata[13]}",
            "<extra></extra>"
        ])
    )

    fig.update_layout(
        title=dict(text="episode", font=dict(size=14), y=0.1, x=0.5),
        xaxis=dict(showticklabels=False, visible=False),
        yaxis_title=emotion
    )

    # print(f'in build_speaker_series_sentiment_hist, df={df}')
    
    return fig
