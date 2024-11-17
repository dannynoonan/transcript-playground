from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator


with DAG('populate_episode_topics', start_date=datetime(2024, 10, 1),
         schedule_interval=None, catchup=False) as dag:
    '''
    Map episodes to topics via vector search, write mappings to es
    '''

    # /populate_all_episode_topics
    populate_ugen_ada002_episode_topics = SimpleHttpOperator(
        task_id='populate_ugen_ada002_episode_topics',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_episode_topics/TNG/universalGenres/openai/ada002',
        method='GET',
        response_filter=lambda response: response.json()['processed_episode_keys'], 
        log_response=True
    )

    populate_ugen_3small_episode_topics = SimpleHttpOperator(
        task_id='populate_ugen_3small_episode_topics',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_episode_topics/TNG/universalGenres/openai/3small',
        method='GET',
        response_filter=lambda response: response.json()['processed_episode_keys'], 
        log_response=True
    )

    # /populate_episode_topic_tfidf_scores
    populate_ugen_ada002_episode_topics_tfidf = SimpleHttpOperator(
        task_id='populate_ugen_ada002_episode_topics_tfidf',
        http_conn_id='tp_api',
        endpoint='esw/populate_episode_topic_tfidf_scores/TNG/universalGenres/openai/ada002',
        method='GET',
        response_filter=lambda response: response.json()['e_keys_to_episode_topics'], 
        log_response=True
    )

    populate_ugen_3small_episode_topics_tfidf = SimpleHttpOperator(
        task_id='populate_ugen_3small_episode_topics_tfidf',
        http_conn_id='tp_api',
        endpoint='esw/populate_episode_topic_tfidf_scores/TNG/universalGenres/openai/3small',
        method='GET',
        response_filter=lambda response: response.json()['e_keys_to_episode_topics'], 
        log_response=True
    )

    populate_ugen_ada002_episode_topics >> populate_ugen_ada002_episode_topics_tfidf >> populate_ugen_3small_episode_topics >> populate_ugen_3small_episode_topics_tfidf
