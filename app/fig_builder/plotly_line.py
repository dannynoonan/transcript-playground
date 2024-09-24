import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app.fig_builder.fig_helper as fh


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


def build_episode_sentiment_line_chart(show_key: str, df: pd.DataFrame, speakers: list, emotions: list, focal_property: str) -> go.Figure:
    print(f'in build_sentiment_line_chart show_key={show_key} emotion={emotions} speakers={speakers} focal_property={focal_property}')

    color_discrete_map = fh.generate_speaker_color_discrete_map(show_key, speakers)

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

    custom_data = ['speaker', 'scene', 'emotion', 'score']

    fig = px.line(df, x='scene', y='score', color=focal_property, height=800, render_mode='svg', line_shape='spline', 
                  custom_data=custom_data, color_discrete_map=color_discrete_map)

    for i in range(len(fig.data)):
        fig.data[i].update(mode='markers+lines')

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{customdata[0]}</b>",
            "Scene %{customdata[1]}",
            "Emotion: %{customdata[2]}",
            "Score: %{customdata[3]:.2f}",
            "<extra></extra>"
        ])
    )    

    return fig
