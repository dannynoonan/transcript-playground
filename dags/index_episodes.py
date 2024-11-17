from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import AirflowException
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor


def _read_response(ti):
    transcripts_es_exists = ti.xcom_pull(
        task_ids='does_transcripts_es_exist',
        key='return_value'
    )
    print(f'transcripts_es_exists={transcripts_es_exists}')
    if not transcripts_es_exists:
        raise AirflowException(f'transcripts_es_exists={transcripts_es_exists}, cannot proceed to index_all_episodes')


# def _branch(ti):
#     transcript_es_exists = ti.xcom_pull(
#         task_ids='does_transcript_es_exist',
#         key='return_value'
#     )
#     print(f'transcript_es_exists={transcript_es_exists}')
#     if transcript_es_exists:
#         return 'index_all_episodes'
#     return 'do_nothing'


# def _do_nothing():
#     print(f'doing nothing')


with DAG('index_episodes', start_date=datetime(2024, 10, 1),
         schedule_interval=None, catchup=False) as dag:
    '''
    Fetch episodes from transcript_db and write to transcripts es index
    '''

    is_api_available = HttpSensor(
        task_id='is_api_available',
        http_conn_id='tp_api',
        endpoint='/',
        method='GET', 
        response_check=lambda response: response.status_code == 200, 
        mode='poke',
    )

    does_transcripts_es_exist = SimpleHttpOperator(
        task_id='does_transcripts_es_exist',
        http_conn_id='tp_api',
        endpoint='/esr/does_index_exist/transcripts',
        method='GET', 
        response_filter=lambda response: response.json()['index_exists'],
        log_response=True
    )

    read_response = PythonOperator(
        task_id='read_response',
        python_callable=_read_response,
    )

    # branch = BranchPythonOperator(
    #     task_id='branch',
    #     python_callable=_branch,
    # )

    # do_nothing = PythonOperator(
    #     task_id='do_nothing',
    #     python_callable=_do_nothing,
    #     # dag=dag
    # )

    index_all_episodes = SimpleHttpOperator(
        task_id='index_all_episodes',
        http_conn_id='tp_api',
        endpoint='esw/index_all_episodes/TNG',
        data={'overwrite_all': 'True'},
        method='GET',
        response_filter=lambda response: response.json()['successful'], # NOTE not actually using this value
        log_response=True
    )

    populate_focal_speakers = SimpleHttpOperator(
        task_id='populate_focal_speakers',
        http_conn_id='tp_api',
        endpoint='esw/populate_focal_speakers/TNG',
        method='GET',
        response_filter=lambda response: response.json()['episodes_to_focal_speakers'], # NOTE not actually using this value
        log_response=True
    )

    populate_focal_locations = SimpleHttpOperator(
        task_id='populate_focal_locations',
        http_conn_id='tp_api',
        endpoint='esw/populate_focal_locations/TNG',
        method='GET',
        response_filter=lambda response: response.json()['episodes_to_focal_locations'], # NOTE not actually using this value
        log_response=True
    )
    
    # is_api_available >> does_transcripts_es_exist >> branch >> [index_all_episodes, do_nothing]

    is_api_available >> does_transcripts_es_exist >> read_response >> index_all_episodes >> populate_focal_speakers >> populate_focal_locations
