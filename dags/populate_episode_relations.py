from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator


with DAG('populate_episode_relations', start_date=datetime(2024, 10, 1),
         schedule_interval=None, catchup=False) as dag:
    '''
    Find most similar episodes using es mlt and embeddings vector search, write similar episode lists to transcripts es 
    '''

    populate_episode_es_mlt_relations = SimpleHttpOperator(
        task_id='populate_episode_es_mlt_relations',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_episode_relations/TNG/es/mlt',
        method='GET',
        response_filter=lambda response: response.json()['episodes_to_relations'], 
        log_response=True
    )

    populate_episode_ada002_relations = SimpleHttpOperator(
        task_id='populate_episode_ada002_relations',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_episode_relations/TNG/openai/ada002',
        method='GET',
        response_filter=lambda response: response.json()['episodes_to_relations'], 
        log_response=True
    )

    populate_episode_3small_relations = SimpleHttpOperator(
        task_id='populate_episode_3small_relations',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_episode_relations/TNG/openai/3small',
        method='GET',
        response_filter=lambda response: response.json()['episodes_to_relations'], 
        log_response=True
    )

    # NOTE can be run in any order 
    populate_episode_es_mlt_relations >> populate_episode_ada002_relations >> populate_episode_3small_relations
