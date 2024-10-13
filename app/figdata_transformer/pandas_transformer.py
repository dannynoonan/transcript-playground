from operator import itemgetter
import pandas as pd



def flatten_topics(topics: list, parent_only: bool = False, max_rank: int = None):
    '''
    This is the simplest topic flattener and ideally everything should be using it
    '''
    out_list = []
    parents_seen = []

    for i, topic in enumerate(topics):
        if max_rank and i >= max_rank:
            break
        t_bits = topic['topic_key'].split('.')
        if len(t_bits) <= 1 or t_bits[0] in parents_seen:
            continue
        parents_seen.append(t_bits[0])
        if parent_only:
            out_list.append(t_bits[0])
        else:
            out_list.append(topic['topic_key'])

    return ', '.join(out_list)


def flatten_and_format_topics_df(df: pd.DataFrame, score_type: str) -> pd.DataFrame:
    '''
    TODO copied after being extracted from another function, not sure where / how this sort of dataframe reformatting should be encapsulated
    '''

    df = df[['topic_key', 'topic_name', 'raw_score', 'score', 'is_parent', 'tfidf_score']]
    df.rename(columns={'score': 'scaled_score'}, inplace=True)
    df['parent_topic'] = df['topic_key'].apply(extract_parent)
    df = df[df['parent_topic'] != df['topic_key']]
    df['total_score'] = df[score_type].sum()
    df.sort_values(score_type, ascending=False, inplace=True)

    return df


def extract_parent(topic_key: str):
    topic_path = topic_key.split('.')
    return topic_path[0]


def flatten_df_list_column(col_contents: list, eligible_col_values: list = None, truncate_at: int = None):
    '''
    Would be a simple list-concat lambda, except that sometimes we want to ignore column values that aren't part of an 'eligible' reference set
    '''
    out_list = []

    i = 0
    for cell_contents in col_contents:
        if truncate_at and i >= truncate_at:
            break
        if eligible_col_values and cell_contents not in eligible_col_values:
            # skip ineligible values without incrementing truncate_at comparison counter
            continue  
        out_list.append(cell_contents)
        i += 1

    return ', '.join(out_list)


def generate_topic_aggs_from_episode_topics(episode_topic_lists: list, max_rank: int = None, max_parent_repeats: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Aggregate topic scores from multiple episodes, output as two dataframes with each topic (parent or leaf) as its own row.
    Takes a list of episode_topics as input. `max_rank` and `max_parent_repeats` inputs for score aggregate tuning.
    '''
    if not max_rank:
        max_rank = 3

    topic_agg_scores = {}
    parent_topic_agg_scores = {}
    topic_agg_tfidf_scores = {}
    parent_topic_agg_tfidf_scores = {}

    for episode_topic_list in episode_topic_lists:
        parents_this_episode = {}
        topic_counter = min(max_rank, len(episode_topic_list))
        for i in range(topic_counter):
            episode_topic = episode_topic_list[i]
            # ignore actual parent topics, do all parent scoring attribution in relation to their child topics
            if episode_topic['is_parent']:
                continue
            topic_key = episode_topic['topic_key']
            parent_topic = topic_key.split('.')[0]
            # aggregate score and ranks for topic
            if topic_key not in topic_agg_scores:
                topic_agg_scores[topic_key] = 0
                topic_agg_tfidf_scores[topic_key] = 0
            topic_agg_scores[topic_key] += episode_topic['score']
            topic_agg_tfidf_scores[topic_key] += episode_topic['tfidf_score']
            # avoid double/triple-scoring using a max_parent_repeats param
            if max_parent_repeats: 
                if parent_topic not in parents_this_episode:
                    parents_this_episode[parent_topic] = 0
                parents_this_episode[parent_topic] += 1
                if parents_this_episode[parent_topic] > max_parent_repeats:
                    continue      
            if parent_topic not in parent_topic_agg_scores:
                parent_topic_agg_scores[parent_topic] = 0
                parent_topic_agg_tfidf_scores[parent_topic] = 0
            parent_topic_agg_scores[parent_topic] += episode_topic['score']
            parent_topic_agg_tfidf_scores[parent_topic] += episode_topic['tfidf_score']

    # topics
    scored_topics = sorted(topic_agg_scores.items(), key=itemgetter(1), reverse=True)
    tfidf_scored_topics = sorted(topic_agg_tfidf_scores.items(), key=itemgetter(1), reverse=True)
    scored_topics_df = pd.DataFrame(scored_topics, columns=['genre', 'score'])
    tfidf_scored_topics_df = pd.DataFrame(tfidf_scored_topics, columns=['genre', 'tfidf_score'])
    topics_df = scored_topics_df.merge(tfidf_scored_topics_df, on='genre')
    topics_df['parent'] = topics_df['genre'].apply(lambda x: x.split('.')[0])
    topics_df.sort_values('parent', inplace=True) # not sure this is needed, or should maybe externalize

    # parent topics
    scored_parent_topics = sorted(parent_topic_agg_scores.items(), key=itemgetter(1), reverse=True)
    tfidf_scored_parent_topics = sorted(parent_topic_agg_tfidf_scores.items(), key=itemgetter(1), reverse=True)
    scored_parent_topics_df = pd.DataFrame(scored_parent_topics, columns=['genre', 'score'])
    tfidf_scored_parent_topics_df = pd.DataFrame(tfidf_scored_parent_topics, columns=['genre', 'tfidf_score'])
    parent_topics_df = scored_parent_topics_df.merge(tfidf_scored_parent_topics_df, on='genre')

    return topics_df, parent_topics_df 
