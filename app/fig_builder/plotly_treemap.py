import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app.fig_builder.fig_helper as fh
from app.show_metadata import TOPIC_COLORS


def build_episode_topic_treemap(df: pd.DataFrame, topic_grouping: str, score_type: str, max_per_parent: int = None) -> go.Figure:
    
    custom_data = ['topic_name', score_type]

    # limit each parent topic to X child topics, avoids appearance of a parent having higher score simply due to having more children
    # NOTE implementing this revealed how completely useless `raw_score` is because all the boxes end up identically sized
    if max_per_parent:
        parents_to_counts = {}
        for index, row in df.iterrows():
            if row['parent_topic'] in parents_to_counts:
                parents_to_counts[row['parent_topic']] += 1
            else:
                parents_to_counts[row['parent_topic']] = 1
            if parents_to_counts[row['parent_topic']] > 3:
                df.drop(index, inplace=True)

    fig = px.treemap(df, path=['parent_topic', 'topic_name'], values=score_type, title=topic_grouping, color='parent_topic',
                     custom_data=custom_data, color_discrete_map=TOPIC_COLORS, width=800, height=650)
    
    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{customdata[0]}</b>",
            "Score: %{customdata[1]:.2f}",
            "<extra></extra>"
        ])
    )

    fig.update_layout(margin=dict(l=30, t=50, r=30, b=30), title_x=0.5)

    return fig
