import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app.figdata_manager.color_meta as cm


def build_topic_aggs_pie(df: pd.DataFrame, topic_grouping: str, score_type: str, is_parent: bool = False) -> go.Figure:

    topic_col = 'genre'
    title = f'{topic_grouping} topic distribution by {score_type}'

    if is_parent:
        color_col='genre'
    else:
        color_col='parent'

    fig = px.pie(df, values=score_type, names=topic_col, title=title, color=color_col, color_discrete_map=cm.TOPIC_COLORS, 
                 height=800, hole=.3)

    fig.update_traces(sort=False, textinfo='label', textposition='inside')

    fig.update_layout(margin=dict(l=30, t=40, r=30, b=30), title_x=0.5)

    return fig
