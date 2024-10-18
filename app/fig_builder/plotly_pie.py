import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app.fig_meta.color_meta as cm
from app import utils


def build_topic_aggs_pie(df: pd.DataFrame, topic_grouping: str, score_type: str, is_parent: bool = False) -> go.Figure:

    topic_col = 'genre'
    title = f'{topic_grouping} topic distribution by {score_type}'

    if is_parent:
        color_col='genre'
    else:
        color_col='parent'

    pie_total = df[score_type].sum()
    df['percentage'] = df[score_type] / pie_total * 100

    fig = px.pie(df, values='percentage', names=topic_col, title=title, color=color_col, color_discrete_map=cm.TOPIC_COLORS, 
                 height=800, hole=.3)

    fig.update_traces(sort=False, textinfo='label', textposition='inside',
                      hovertemplate="%{label} (%{value:.2f}%)")

    fig.update_layout(margin=dict(l=30, t=40, r=30, b=30), title_x=0.5)

    return fig
