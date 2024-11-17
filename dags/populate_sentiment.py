from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

import app.data_service.sentiment_populator as sp


def _populate_episode_sentiment(ti):
    show_key = 'TNG'
    analyzer = 'openai_emo'
    epiode_keys = ti.xcom_pull(task_ids='fetch_episode_keys')
    for e_key in epiode_keys:
        sp.populate_episode_sentiment(show_key, e_key, analyzer, scene_level=True, overwrite_csv=True)


with DAG('populate_sentiment', start_date=datetime(2024, 10, 1),
         schedule_interval=None, catchup=False) as dag:
    '''
    Populate sentiment
    '''

    fetch_episode_keys = SimpleHttpOperator(
        task_id='fetch_episode_keys',
        http_conn_id='tp_api',
        endpoint='esr/fetch_simple_episodes/TNG',
        method='GET',
        response_filter=lambda response: [episode['episode_key'] for episode in response.json()['episodes']], 
        log_response=True
    )

    populate_series_sentiment = PythonOperator(
        task_id='populate_series_sentiment',
        python_callable=_populate_episode_sentiment
    )
    
    fetch_episode_keys >> populate_series_sentiment
