import plotly.graph_objects as go


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
        