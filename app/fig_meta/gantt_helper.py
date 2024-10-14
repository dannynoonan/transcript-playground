import plotly.graph_objects as go


def simple_season_episode_i_map(episodes_by_season: dict):
    season_episode_i_map = {}
    episode_i = 0
    for season, episodes in episodes_by_season.items():
        season_episode_i_map[season] = episode_i
        episode_i += len(episodes)

    return season_episode_i_map


def build_and_annotate_scene_labels(fig: go.Figure, scenes: list) -> list:
    """
    Helper function to layer scene labels into episode dialog gantt
    """

    # build markers and labels marking events 
    scene_lines = []
    yshift = -22 # NOTE might need to be derived based on speaker count / y axis length

    for scene in scenes:
        # add vertical line for each scene
        scene_line = dict(type='line', line_width=1, line_color='#A0A0A0', x0=scene['Start'], x1=scene['Start'], y0=0, y1=1, yref='paper')
        scene_lines.append(scene_line)
        # add annotation for each scene location
        fig.add_annotation(x=scene['Start'], y=0, text=scene['Task'], showarrow=False, 
            yshift=yshift, xshift=6, textangle=-90, align='left', yanchor='bottom',
            font=dict(family="Arial", size=10, color="#A0A0A0"))

    return scene_lines


def build_and_annotate_season_labels(fig: go.Figure, seasons_to_first_episodes: dict) -> list:
    """
    Helper function to layer season labels into series continuity gantts
    """

    # build markers and labels marking events 
    season_lines = []
    yshift = -22 # NOTE might need to be derived based on season count / y axis length

    for season, episode_i in seasons_to_first_episodes.items():
        # add vertical line for each season
        season_line = dict(type='line', line_width=1, line_color='#A0A0A0', x0=episode_i, x1=episode_i, y0=0, y1=1, yref='paper')
        season_lines.append(season_line)
        # add annotation for each season label
        fig.add_annotation(x=episode_i, y=0, text=f'Season {season}', showarrow=False, 
            yshift=yshift, xshift=6, textangle=-90, align='left', yanchor='bottom',
            font=dict(family="Arial", size=10, color="#A0A0A0"))

    return season_lines