from dash import dash_table
import pandas as pd

import app.fig_meta.color_meta as cm
import app.page_builder_service.page_components as pc


def generate_bertopic_model_clusters_dt(show_key: str, clusters_df: pd.DataFrame) -> dash_table.DataTable:
    display_cols = ['cluster', 'cluster_title', 'speaker_group', 'probability', 'wc', 'season', 'episode', 'episode_title', 'topics']
    
    # cluster_list = [str(c) for c in list(clusters_df['cluster'].unique())]
    cluster_list = list(clusters_df['cluster'].unique())

    # reformat columns, sort table
    clusters_df['air_date'] = clusters_df['air_date'].apply(lambda x: x[:10])
    # clusters_df['episode_key'] = clusters_df['episode_key'].apply(lambda x: str(x))
    # clusters_df['topics'] = clusters_df['topics'].apply(lambda x: ", ".join(x))
    # clusters_df['focal_speakers'] = clusters_df['focal_speakers'].apply(lambda x: ", ".join(x))
    # clusters_df['focal_locations'] = clusters_df['focal_locations'].apply(lambda x: ", ".join(x))
    clusters_df['episode_title'] = clusters_df.apply(lambda x: pc.link_to_episode(show_key, x['episode_key'], x['episode_title']), axis=1)
    # clusters_df.sort_values(['cluster', 'season', 'episode', 'probability'], ascending=[True, True, True, False], inplace=True)

    # define inputs for df->dt conversion
    colors = list(cm.BGCOLORS_TO_TEXT_COLORS.keys())
    numeric_precision_overrides = {'probability': 2}
    # bg_color_map = {f'cluster_{i}':c for i, c in enumerate(colors)}
    bg_color_map = {str(i):c for i, c in enumerate(colors)}
    clusters_dt = pc.pandas_df_to_dash_dt(clusters_df, display_cols, 'cluster', cluster_list, bg_color_map,
                                          numeric_precision_overrides=numeric_precision_overrides,
                                          md_cols=['episode_title'])                                              

    return clusters_dt
