from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator


with DAG('populate_speaker_topics', start_date=datetime(2024, 10, 1),
         schedule_interval=None, catchup=False) as dag:
    '''
    Map speakers to topics via vector search, write mappings to es
    '''

    # mbti
    populate_mbti_ada002_speaker_topics = SimpleHttpOperator(
        task_id='populate_mbti_ada002_speaker_topics',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_speaker_topics/TNG/mbti/openai/ada002',
        method='GET',
        response_filter=lambda response: response.json()['success_count'], 
        log_response=True
    )

    populate_mbti_3small_speaker_topics = SimpleHttpOperator(
        task_id='populate_mbti_3small_speaker_topics',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_speaker_topics/TNG/mbti/openai/3small',
        method='GET',
        response_filter=lambda response: response.json()['success_count'], 
        log_response=True
    )

    # dndAlignments
    populate_dnda_ada002_speaker_topics = SimpleHttpOperator(
        task_id='populate_dnda_ada002_speaker_topics',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_speaker_topics/TNG/dndAlignments/openai/ada002',
        method='GET',
        response_filter=lambda response: response.json()['success_count'], 
        log_response=True
    )

    populate_dnda_3small_speaker_topics = SimpleHttpOperator(
        task_id='populate_dnda_3small_speaker_topics',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_speaker_topics/TNG/dndAlignments/openai/3small',
        method='GET',
        response_filter=lambda response: response.json()['success_count'], 
        log_response=True
    )

    # NOTE can be run in any order 
    populate_mbti_ada002_speaker_topics >> populate_mbti_3small_speaker_topics >> populate_dnda_ada002_speaker_topics >> populate_dnda_3small_speaker_topics
