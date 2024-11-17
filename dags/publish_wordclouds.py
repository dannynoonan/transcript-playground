from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

import app.data_service.wordcloud_publisher as wp


def _publish_series_wordcloud():
    show_key = 'TNG'
    max_words = 50
    wp.publish_series_wordcloud(show_key, max_words=max_words)


def _publish_season_wordclouds(ti):
    show_key = 'TNG'
    max_words = 50
    seasons = ti.xcom_pull(task_ids='fetch_seasons')
    wp.publish_season_wordclouds(show_key, seasons, max_words=max_words)


def _publish_episode_wordclouds(ti):
    show_key = 'TNG'
    max_words = 50
    epiode_keys = ti.xcom_pull(task_ids='fetch_episode_keys')
    wp.publish_episode_wordclouds(show_key, epiode_keys, max_words=max_words)


with DAG('publish_wordclouds', start_date=datetime(2024, 10, 1),
         schedule_interval=None, catchup=False) as dag:
    '''
    Publish wordclouds for all seasons and episodes of a series
    '''

    publish_series_wordcloud = PythonOperator(
        task_id='publish_series_wordcloud',
        python_callable=_publish_series_wordcloud
    )

    fetch_seasons = SimpleHttpOperator(
        task_id='fetch_seasons',
        http_conn_id='tp_api',
        endpoint='esr/list_seasons/TNG',
        method='GET',
        response_filter=lambda response: [season for season in response.json()['seasons']], 
        log_response=True
    )

    publish_season_wordclouds = PythonOperator(
        task_id='publish_season_wordclouds',
        python_callable=_publish_season_wordclouds
    )

    fetch_episode_keys = SimpleHttpOperator(
        task_id='fetch_episode_keys',
        http_conn_id='tp_api',
        endpoint='esr/fetch_simple_episodes/TNG',
        method='GET',
        response_filter=lambda response: [episode['episode_key'] for episode in response.json()['episodes']], 
        log_response=True
    )

    publish_episode_wordclouds = PythonOperator(
        task_id='publish_episode_wordclouds',
        python_callable=_publish_episode_wordclouds
    )
    
    publish_series_wordcloud >> fetch_seasons >> publish_season_wordclouds >> fetch_episode_keys >> publish_episode_wordclouds
