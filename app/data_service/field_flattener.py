import numpy as np
from operator import itemgetter
import pandas as pd

import app.data_service.calculator as calc
import app.data_service.matrix_operations as mxop


def flatten_es_topics(topics: list) -> list:
    simple_topics = []
    count = 0
    for t in topics:
        if 'is_parent' in t and t['is_parent']:
            continue
        simple_topic = dict(topic_key=t['topic_key'], topic_name=t['topic_name'], score=t['score'], raw_score=t['raw_score'])
        if 'tfidf_score' in t:
            simple_topic['tfidf_score'] = t['tfidf_score']
        simple_topics.append(simple_topic)
        count += 1
        if count > 5:
            break
    return simple_topics


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
                flat_spkr['scaled_score'] = calc.normalize_score(topic['raw_score'], percent_distrib_list)
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
