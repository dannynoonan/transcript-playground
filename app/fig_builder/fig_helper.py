import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from app import utils


FRAME_RATE = 1000


def apply_animation_settings(fig: go.Figure, base_fig_title: str, frame_rate: int = None) -> None:
    """
    generic recipe of steps to execute on animation figure after its built: explicitly set frame rate, dynamically update fig title, etc
    """

    print(f'in apply_animation_settings frame_rate={frame_rate}')
    if not frame_rate:
        frame_rate = FRAME_RATE

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_rate
        
    for button in fig.layout.updatemenus[0].buttons:
        button["args"][1]["frame"]["redraw"] = True
    
    # first_step = True
    for step in fig.layout.sliders[0].steps:
        step["args"][1]["frame"]["redraw"] = True
        # if first_step:
        #     # step["args"][1]["frame"]["duration"] = 0
        #     first_step = False
        # step["args"][1]["frame"]["duration"] = frame_rate

    # for k in range(len(fig.frames)):
    #     year = YEAR_0 + (k*4)
    #     era = get_era_for_year(year)
    #     fig.frames[k]['layout'].update(title_text=f'{base_fig_title}: {year} ({era})')
        
    print(f'fig.layout={fig.layout}')


# def flatten_focal_speakers(speakers: list, eligible_speakers: list = None, max_rank: int = None):
#     '''
#     Could be a simple lambda function, but sometimes we want to ignore focal speakers that aren't part of the regular or recurring cast
#     '''
#     out_list = []

#     i = 0
#     for spkr in speakers:
#         if max_rank and i >= max_rank:
#             break
#         if eligible_speakers and spkr not in eligible_speakers:
#             # skip without incrementing max_rank comparison counter
#             continue  
#         out_list.append(spkr)
#         i += 1

#     return ', '.join(out_list)


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


def scale_values(values: list, low: int = 0, high: int = 1) -> list:
    raw_low = np.min(values)
    raw_high = np.max(values)
    raw_range = raw_high - raw_low
    scaled_range = high - low
    scaled_values = []
    for v in values:
        scaled_v = (v - raw_low) / raw_range * scaled_range + low
        scaled_values.append(scaled_v)
    return scaled_values


def matplotlib_gradient_to_rgb_strings(gradient_type: str):
    gradient = plt.cm.get_cmap(gradient_type)
    rgb_strings = []
    for c in gradient.colors:
        mpl_color = matplotlib.colors.to_rgb(c)
        rgb = tuple([int(c*255) for c in mpl_color])
        rgb_str = f'rgb{rgb}'.replace(' ', '')
        rgb_strings.append(rgb_str)
    return rgb_strings


def map_range_values_to_gradient(range_values: list, gradient_values: list) -> list:
    '''
    Both 'range_values' and 'gradient_values' are assumed to be sorted
    '''
    scaled_range_values = scale_values(range_values, 0, len(gradient_values)-1)
    discrete_gradient_values = []
    for v in scaled_range_values:
        discrete_gradient_values.append(gradient_values[round(v)])

    # TODO incorporate Black vs White font color here?

    return discrete_gradient_values


def explode_speaker_topics(speakers: list, topic_type: str, limit_per_speaker: int = None, percent_distrib_list: list = None) -> list: 
    '''
    Expand individual speaker rows containing multiple nested topics into multiple speaker rows each containing one topic
    '''

    if not limit_per_speaker:
        limit_per_speaker = 10
    topic_field = f'topics_{topic_type}'
    
    exploded_speakers = []
    for spkr in speakers:
        if topic_field not in spkr or (spkr['word_count'] < 20 and spkr['line_count'] < 3):
            continue
        
        # extract each topic (up to topic_limit) into its own flattened speaker item
        topic_limit = min(limit_per_speaker, len(spkr[topic_field]))
        for i in range(topic_limit):
            flat_spkr = spkr.copy()
            exploded_speakers.append(flat_spkr)
            topic = spkr[topic_field][i]
            flat_spkr['topic_key'] = topic['topic_key']
            flat_spkr['topic_name'] = topic['topic_name']
            flat_spkr['rank'] = i+1
            flat_spkr['dot_size'] = (topic_limit - i) / topic_limit
            flat_spkr['score'] = topic['score']
            flat_spkr['raw_score'] = topic['raw_score']
            # NOTE sad kazoo this was conceived on a false premise, but keeping it in here for now
            if percent_distrib_list:
                flat_spkr['scaled_score'] = utils.normalize_score(topic['raw_score'], percent_distrib_list)
            # extract each topic (up to topic_limit) into its own flattened speaker item
            del flat_spkr[topic_field]

    return exploded_speakers


def flatten_speaker_topics(speakers: list, topic_type: str, limit_per_speaker: int = None) -> list: 
    '''
    Replace nested speaker topic dicts with concatenated string of topic_keys in speaker rows, dropping speakers with few lines/words in process
    For reasons I can't recall, I'm being careful to copy each speaker rather than altering the existing speakers
    '''

    if not limit_per_speaker:
        limit_per_speaker = 10
    topic_field = f'topics_{topic_type}'
    
    flattened_speakers = []
    for spkr in speakers:
        if spkr['word_count'] < 20 and spkr['line_count'] < 3:
            continue

        flat_spkr = spkr.copy()
        topics = []
        if topic_field in spkr:
            # extract each topic (up to topic_limit) into its own flattened speaker item
            topic_limit = min(limit_per_speaker, len(spkr[topic_field]))
            for i in range(topic_limit):
                topic = spkr[topic_field][i]
                topics.append(topic['topic_key'])

        flat_spkr[topic_field] = ', '.join(topics)
        flattened_speakers.append(flat_spkr)

    return flattened_speakers


def flatten_and_refine_alt_names(speakers: list, ignore_dupes: bool = False, limit_per_speaker: int = None) -> list:
    flattened_speakers = []
    for spkr in speakers:
        flat_spkr = spkr.copy()
        alt_names = []
        if 'alt_names' in spkr:
            alt_names_limit = min(limit_per_speaker, len(spkr['alt_names']))
            alt_names = []
            for i in range(alt_names_limit):
                if ignore_dupes and spkr['alt_names'][i].upper() == spkr['speaker'].upper():
                    continue
                alt_names.append(spkr['alt_names'][i])
        flat_spkr['aka'] = ', '.join(alt_names)
        flattened_speakers.append(flat_spkr)

    return flattened_speakers


def blank_fig():
    '''
    Best I've come up with so far to dynamically show or hide (almost) a graph that's declared as a dash page object
    '''
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None, width=10, height=10)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    
    return fig


# NOTE not using and not sure if will be used
# def generate_topic_aggs_from_episodes(episodes: list, topic_type: str, max_rank: int = None, max_parent_repeats: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
#     '''
#     Aggregate topic scores from multiple episodes, output as two dataframes with each topic (parent or leaf) as its own row.
#     Takes a list of episodes (which may have summarized/truncated topics) as input. `max_rank` and `max_parent_repeats` inputs for score aggregate tuning.
#     '''
#     if not max_rank:
#         max_rank = 3

#     topic_agg_scores = {}
#     parent_topic_agg_scores = {}
#     topic_agg_tfidf_scores = {}
#     parent_topic_agg_tfidf_scores = {}
        
#     for episode in episodes:
#         if topic_type not in episode:
#             continue
#         parents_this_episode = {}
#         topics = episode[topic_type]
#         topic_counter = min(max_rank, len(topics))
#         for i in range(topic_counter):
#             topic = topics[i]
#             topic_key = topics[i]['topic_key']
#             parent_topic = topic_key.split('.')[0]
#             # aggregate score and ranks for topic
#             if topic_key not in topic_agg_scores:
#                 topic_agg_scores[topic_key] = 0
#                 topic_agg_tfidf_scores[topic_key] = 0
#             topic_agg_scores[topic_key] += topic['score']
#             topic_agg_tfidf_scores[topic_key] += topic['tfidf_score']
#             # avoid double/triple-scoring using a max_parent_repeats param
#             if max_parent_repeats: 
#                 if parent_topic not in parents_this_episode:
#                     parents_this_episode[parent_topic] = 0
#                 parents_this_episode[parent_topic] += 1
#                 if parents_this_episode[parent_topic] > max_parent_repeats:
#                     continue      
#             if parent_topic not in parent_topic_agg_scores:
#                 parent_topic_agg_scores[parent_topic] = 0
#                 parent_topic_agg_tfidf_scores[parent_topic] = 0
#             parent_topic_agg_scores[parent_topic] += topic['score']
#             parent_topic_agg_tfidf_scores[parent_topic] += topic['tfidf_score']

#     # topics
#     scored_topics = sorted(topic_agg_scores.items(), key=itemgetter(1), reverse=True)
#     tfidf_scored_topics = sorted(topic_agg_tfidf_scores.items(), key=itemgetter(1), reverse=True)
#     scored_topics_df = pd.DataFrame(scored_topics, columns=['genre', 'score'])
#     tfidf_scored_topics_df = pd.DataFrame(tfidf_scored_topics, columns=['genre', 'tfidf_score'])
#     topics_df = scored_topics_df.merge(tfidf_scored_topics_df, on='genre')
#     topics_df['parent'] = topics_df['genre'].apply(lambda x: x.split('.')[0])
#     topics_df.sort_values('parent', inplace=True) # not sure this is needed, or should maybe externalize

#     # parent topics
#     scored_parent_topics = sorted(parent_topic_agg_scores.items(), key=itemgetter(1), reverse=True)
#     tfidf_scored_parent_topics = sorted(parent_topic_agg_tfidf_scores.items(), key=itemgetter(1), reverse=True)
#     scored_parent_topics_df = pd.DataFrame(scored_parent_topics, columns=['genre', 'score'])
#     tfidf_scored_parent_topics_df = pd.DataFrame(tfidf_scored_parent_topics, columns=['genre', 'tfidf_score'])
#     parent_topics_df = scored_parent_topics_df.merge(tfidf_scored_parent_topics_df, on='genre')

#     return topics_df, parent_topics_df 
        