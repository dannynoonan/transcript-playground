import numpy as np
from operator import itemgetter
import pandas as pd

import app.figdata_manager.matrix_operations as mxop
from app import utils


def flatten_and_format_topics_df(df: pd.DataFrame, score_type: str) -> pd.DataFrame:
    '''
    TODO copied after being extracted from another function, not sure where / how this sort of dataframe reformatting should be encapsulated
    '''

    df = df[['topic_key', 'topic_name', 'raw_score', 'score', 'is_parent', 'tfidf_score']]
    df.rename(columns={'score': 'scaled_score'}, inplace=True)
    df['parent_topic'] = df['topic_key'].apply(mxop.extract_parent)
    df = df[df['parent_topic'] != df['topic_key']]
    df['total_score'] = df[score_type].sum()
    df.sort_values(score_type, ascending=False, inplace=True)

    return df


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
