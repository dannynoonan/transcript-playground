from datetime import datetime
import json
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import DAG
# from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor


with DAG('load_episodes', start_date=datetime(2024, 10, 1),
         schedule_interval='@daily', catchup=False) as dag:
    '''
    Load series listing and transcript source metadata, followed by episode transcript data, into transcript_db
    '''

    is_api_available = HttpSensor(
        task_id='is_api_available',
        http_conn_id='tp_api',
        endpoint='/',
        method='GET', 
        response_check=lambda response: response.status_code == 200, 
        mode='poke',
    )

    load_episode_listing = SimpleHttpOperator(
        task_id='load_episode_listing',
        http_conn_id='tp_api',
        endpoint='etl/load_episode_listing/TNG',
        data={'write_to_db': 'True'},
        method='GET',
        response_filter=lambda response: response.json()['episode_count'],
        log_response=True
    )

    load_transcript_sources = SimpleHttpOperator(
        task_id='load_transcript_sources',
        http_conn_id='tp_api',
        endpoint='etl/load_transcript_sources/TNG',
        data={'write_to_db': 'True'},
        method='GET',
        response_filter=lambda response: response.json()['transcript_sources_count'],
        log_response=True
    )

    load_all_transcripts = SimpleHttpOperator(
        task_id='load_all_transcripts',
        http_conn_id='tp_api',
        endpoint='etl/load_all_transcripts/TNG',
        data={'overwrite_all': 'True'},
        method='GET',
        response_filter=lambda response: response.json()['successful'],
        log_response=True
    )

    # fetch_psql_episode = PythonOperator(
    #     task_id='fetch_psql_episode',
    #     python_callable=_fetch_psql_episode
    # )
    
    # write_psql_episode_to_es = PythonOperator(
    #     task_id='write_psql_episode_to_es',
    #     python_callable=_write_psql_episode_to_es
    # )
    
    is_api_available >> load_episode_listing >> load_transcript_sources >> load_all_transcripts
