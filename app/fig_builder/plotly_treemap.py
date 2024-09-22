import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app.fig_builder.fig_helper as fh
from app.show_metadata import TOPIC_COLORS


def build_episode_topic_treemap(df: pd.DataFrame, topic_grouping: str, score_type: str) -> go.Figure:
    df = df[['topic_key', 'topic_name', 'raw_score', 'score', 'is_parent', 'tfidf_score']]
    df['parent_topic'] = df['topic_key'].apply(fh.extract_parent)
    df = df[df['parent_topic'] != df['topic_key']]
    df['total_score'] = df[score_type].sum()

    fig = px.treemap(df, path=['parent_topic', 'topic_name'], values=score_type, title=topic_grouping, color='parent_topic',
                     color_discrete_map=TOPIC_COLORS, width=800, height=650)

    return fig
