from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator


with DAG('copy_sources', start_date=datetime(2024, 10, 1),
         schedule_interval=None, catchup=False) as dag:
    '''
    Copy episode listing metadata, transcript source metadata, and transcript text to local txt files 
    '''

    copy_episode_listing = SimpleHttpOperator(
        task_id='copy_episode_listing',
        http_conn_id='tp_api',
        endpoint='etl/copy_episode_listing/TNG',
        method='GET',
        response_filter=lambda response: response.json()['file_path'], 
        log_response=True
    )

    copy_transcript_sources = SimpleHttpOperator(
        task_id='copy_transcript_sources',
        http_conn_id='tp_api',
        endpoint='etl/copy_transcript_sources/TNG',
        method='GET',
        response_filter=lambda response: response.json()['file_path'], 
        log_response=True
    )

    copy_transcripts_from_source = SimpleHttpOperator(
        task_id='copy_transcripts_from_source',
        http_conn_id='tp_api',
        endpoint='etl/copy_all_transcripts_from_source/TNG',
        method='GET',
        response_filter=lambda response: response.json()['successful_episode_keys'], 
        log_response=True
    )
    
    copy_episode_listing >> copy_transcript_sources >> copy_transcripts_from_source
