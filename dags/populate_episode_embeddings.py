from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

# from airflow import AirflowException
from airflow import DAG
# from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

# import app.es.es_write_router as esw
# from app.show_metadata import ShowKey


with DAG('populate_episode_embeddings', start_date=datetime(2024, 10, 1),
         schedule_interval=None, catchup=False) as dag:
    '''
    Generate episode embeddings and write to transcripts es
    '''

    # fetch_episode_keys = SimpleHttpOperator(
    #     task_id='fetch_episode_keys',
    #     http_conn_id='tp_api',
    #     endpoint='esr/fetch_simple_episodes/TNG',
    #     method='GET',
    #     response_filter=lambda response: [episode['episode_key'] for episode in response.json()['episodes']], 
    #     log_response=True
    # )

    populate_episode_ada002_embeddings = SimpleHttpOperator(
        task_id='populate_episode_ada002_embeddings',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_episode_embeddings/TNG/openai/ada002',
        method='GET',
        response_filter=lambda response: response.json()['processed_episode_keys'], 
        log_response=True
    )

    populate_episode_3small_embeddings = SimpleHttpOperator(
        task_id='populate_episode_3small_embeddings',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_episode_embeddings/TNG/openai/3small',
        method='GET',
        response_filter=lambda response: response.json()['processed_episode_keys'], 
        log_response=True
    )

    # NOTE can be run in any order 
    populate_episode_ada002_embeddings >> populate_episode_3small_embeddings
