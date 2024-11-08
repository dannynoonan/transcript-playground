from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator


with DAG('index_topics', start_date=datetime(2024, 10, 1),
         schedule_interval=None, catchup=False) as dag:
    '''
    Load topic data from csv into es, generate embeddings for each topic's description
    '''

    # load all topics in a given topic group from csv into es

    index_mbti_topic_grouping = SimpleHttpOperator(
        task_id='index_mbti_topic_grouping',
        http_conn_id='tp_api',
        endpoint='esw/index_topic_grouping/mbti',
        method='GET',
        response_filter=lambda response: response.json()['successful_topics'], 
        log_response=True
    )

    index_dnda_topic_grouping = SimpleHttpOperator(
        task_id='index_dnda_topic_grouping',
        http_conn_id='tp_api',
        endpoint='esw/index_topic_grouping/dndAlignments',
        method='GET',
        response_filter=lambda response: response.json()['successful_topics'], 
        log_response=True
    )

    index_univeral_genres_topic_grouping = SimpleHttpOperator(
        task_id='index_univeral_genres_topic_grouping',
        http_conn_id='tp_api',
        endpoint='esw/index_topic_grouping/universalGenres',
        method='GET',
        response_filter=lambda response: response.json()['successful_topics'], 
        log_response=True
    )

    # populate embeddings for all topics in a given topic group for a given transformer model

    # mbti
    populate_mbti_topic_ada002_embeddings = SimpleHttpOperator(
        task_id='populate_mbti_topic_ada002_embeddings',
        http_conn_id='tp_api',
        endpoint='esw/populate_topic_grouping_embeddings/mbti/openai/ada002',
        method='GET',
        response_filter=lambda response: response.json()['successful_topics'], 
        log_response=True
    )

    populate_mbti_topic_3small_embeddings = SimpleHttpOperator(
        task_id='populate_mbti_topic_3small_embeddings',
        http_conn_id='tp_api',
        endpoint='esw/populate_topic_grouping_embeddings/mbti/openai/3small',
        method='GET',
        response_filter=lambda response: response.json()['successful_topics'], 
        log_response=True
    )

    # dndAlignments
    populate_dnda_topic_ada002_embeddings = SimpleHttpOperator(
        task_id='populate_dnda_topic_ada002_embeddings',
        http_conn_id='tp_api',
        endpoint='esw/populate_topic_grouping_embeddings/dndAlignments/openai/ada002',
        method='GET',
        response_filter=lambda response: response.json()['successful_topics'], 
        log_response=True
    )

    populate_dnda_topic_3small_embeddings = SimpleHttpOperator(
        task_id='populate_dnda_topic_3small_embeddings',
        http_conn_id='tp_api',
        endpoint='esw/populate_topic_grouping_embeddings/dndAlignments/openai/3small',
        method='GET',
        response_filter=lambda response: response.json()['successful_topics'], 
        log_response=True
    )

    # universalGenres
    populate_universal_genres_topic_ada002_embeddings = SimpleHttpOperator(
        task_id='populate_universal_genres_topic_ada002_embeddings',
        http_conn_id='tp_api',
        endpoint='esw/populate_topic_grouping_embeddings/universalGenres/openai/ada002',
        method='GET',
        response_filter=lambda response: response.json()['successful_topics'], 
        log_response=True
    )

    populate_universal_genres_topic_3small_embeddings = SimpleHttpOperator(
        task_id='populate_universal_genres_topic_3small_embeddings',
        http_conn_id='tp_api',
        endpoint='esw/populate_topic_grouping_embeddings/universalGenres/openai/3small',
        method='GET',
        response_filter=lambda response: response.json()['successful_topics'], 
        log_response=True
    )

    index_mbti_topic_grouping >> populate_mbti_topic_ada002_embeddings >> populate_mbti_topic_3small_embeddings >> index_dnda_topic_grouping >> populate_dnda_topic_ada002_embeddings >> populate_dnda_topic_3small_embeddings >> index_univeral_genres_topic_grouping >> populate_universal_genres_topic_ada002_embeddings >> populate_universal_genres_topic_3small_embeddings
