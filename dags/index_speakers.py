from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import AirflowException
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator


def _check_responses(ti):
    index_exists = ti.xcom_pull(
        task_ids='verify_speakers_es',
        key='return_value'
    )
    print(f'verify_speakers_es={index_exists}')
    if not index_exists:
        raise AirflowException(f'speakers es index not found, cannot proceed to index_all_speakers')
    
    index_exists = ti.xcom_pull(
        task_ids='verify_speaker_episodes_es',
        key='return_value'
    )
    print(f'verify_speaker_episodes_es={index_exists}')
    if not index_exists:
        raise AirflowException(f'speaker_episodes es index not found, cannot proceed to index_all_speakers')
    
    index_exists = ti.xcom_pull(
        task_ids='verify_speaker_seasons_es',
        key='return_value'
    )
    print(f'verify_speaker_seasons_es={index_exists}')
    if not index_exists:
        raise AirflowException(f'speaker_seasons es index not found, cannot proceed to index_all_speakers')


with DAG('index_speakers', start_date=datetime(2024, 10, 1),
         schedule_interval='@daily', catchup=False) as dag:
    '''
    Load speaker metadata from files and write to speakers es index
    '''

    verify_speakers_es = SimpleHttpOperator(
        task_id='verify_speakers_es',
        http_conn_id='tp_api',
        endpoint='/esr/does_index_exist/speakers',
        method='GET', 
        response_filter=lambda response: response.json()['index_exists'],
        log_response=True
    )

    verify_speaker_episodes_es = SimpleHttpOperator(
        task_id='verify_speaker_episodes_es',
        http_conn_id='tp_api',
        endpoint='/esr/does_index_exist/speaker_episodes',
        method='GET', 
        response_filter=lambda response: response.json()['index_exists'],
        log_response=True
    )

    verify_speaker_seasons_es = SimpleHttpOperator(
        task_id='verify_speaker_seasons_es',
        http_conn_id='tp_api',
        endpoint='/esr/does_index_exist/speaker_seasons',
        method='GET', 
        response_filter=lambda response: response.json()['index_exists'],
        log_response=True
    )

    check_responses = PythonOperator(
        task_id='check_responses',
        python_callable=_check_responses,
    )

    index_all_speakers = SimpleHttpOperator(
        task_id='index_all_speakers',
        http_conn_id='tp_api',
        endpoint='esw/index_all_speakers/TNG',
        method='GET',
        response_filter=lambda response: response.json()['successful'], # NOTE not actually using this value
        log_response=True
    )

    [verify_speakers_es, verify_speaker_episodes_es, verify_speaker_seasons_es] >> check_responses >> index_all_speakers
