from dash import dash_table
import pandas as pd

import app.fig_meta.color_meta as cm
import app.page_builder_service.page_components as pc


def generate_episode_dropdown_options(show_key: str, all_episodes: list) -> list:
    episode_dropdown_options = []
    for episode in all_episodes:
        label = f"{episode['title']} (S{episode['season']}:E{episode['sequence_in_season']})"
        episode_dropdown_options.append({'label': label, 'value': episode['episode_key']})
    
    return episode_dropdown_options


def generate_similar_episodes_df(mlt_matches: list, all_episodes: list, episode_key: str, mlt_type: str) -> pd.DataFrame:
    '''
    TODO
    '''
    all_episodes = [dict(episode, rank=len(mlt_matches)+1, rev_rank=1, score=0, group='all') for episode in all_episodes]
    all_episodes_dict = {episode['episode_key']:episode for episode in all_episodes}

    # transfer rank/score properties from mlt_matches to all_episodes_dict
    for i, mlt_match in enumerate(mlt_matches):
        if mlt_match['episode_key'] in all_episodes_dict:
            ep = all_episodes_dict[mlt_match['episode_key']]
            if mlt_type == 'openai_embeddings':
                ep['rank'] = i+1
            else:
                ep['rank'] = mlt_match['rank']
            ep['rev_rank'] = len(mlt_matches) - ep['rank']
            ep['score'] = mlt_match['score']
            ep['group'] = 'mlt'

    # assign 'highest' rank/score properties to focal episode inside all_episodes_dict
    high_score = mlt_matches[0]['score']
    all_episodes_dict[episode_key]['rank'] = 0
    all_episodes_dict[episode_key]['rev_rank'] = len(mlt_matches) + 1
    all_episodes_dict[episode_key]['score'] = high_score + .01
    all_episodes_dict[episode_key]['group'] = 'focal'

    all_episodes = list(all_episodes_dict.values())

    # load all episodes, with ranks/scores assigned to focal episode and similar episodes, into dataframe
    df = pd.DataFrame(all_episodes)

    return df


def generate_episode_search_results_dt(show_key: str, matching_lines_df: pd.DataFrame) -> dash_table.DataTable:
    '''    
    Early attempt to extract the guts of dash datatable construction out of callback modules
    '''
    # rename columns for display
    matching_lines_df.rename(columns={'Task': 'character', 'scene_event': 'line', 'Line': 'dialog'}, inplace=True)

    # define inputs for df->dt conversion
    matching_speakers = list(matching_lines_df['character'].unique())
    speaker_color_map = cm.generate_speaker_color_discrete_map(show_key, matching_speakers)
    display_cols = ['character', 'scene', 'line', 'location', 'dialog']
    episode_search_results_dt = pc.pandas_df_to_dash_dt(matching_lines_df, display_cols, 'character', matching_speakers, speaker_color_map,
                                                        numeric_precision_overrides={'scene': 0, 'line': 0}, md_cols=['dialog'])
    
    return episode_search_results_dt


def generate_episode_similarity_dt(show_key: str, df: pd.DataFrame) -> dash_table.DataTable:
    '''
    Early attempt to extract the guts of dash datatable construction out of callback modules
    '''
    # vector operations on columns to generate presentation data
    df['title'] = df.apply(lambda x: pc.link_to_episode(show_key, x['episode_key'], x['title']), axis=1)
    
    # drop / sort by rows for display
    df = df.loc[df['score'] > 0]
    df = df.loc[df['rank'] > 0]
    df.sort_values('rank', inplace=True, ascending=True)
    
    # define inputs for df->dt conversion
    similar_episode_scores = list(df['score'].values)
    viridis_discrete_rgbs = cm.matplotlib_gradient_to_rgb_strings('viridis')
    sim_ep_rgbs = cm.map_range_values_to_gradient(similar_episode_scores, viridis_discrete_rgbs)
    # sim_ep_rgb_textcolors = {rgb:"Black" for rgb in sim_ep_rgbs}
    display_cols = ['title', 'season', 'episode', 'score', 'rank', 'flattened_topics']
    episode_similarity_dt = pc.pandas_df_to_dash_dt(df, display_cols, 'rank', sim_ep_rgbs, {}, numeric_precision_overrides={'score': 2}, md_cols=['title'])

    return episode_similarity_dt
