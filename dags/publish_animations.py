from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

import app.data_service.animation_publisher as ap


def _publish_series_speaker_frequency_bars():
    show_key = 'TNG'
    fig_type = 'speaker_frequency_bar'
    span_granularities = ['word', 'line', 'scene']
    for span_gran in span_granularities:
        ap.publish_animation(show_key, fig_type, span_granularity=span_gran)

def _publish_season_speaker_frequency_bars(ti):
    show_key = 'TNG'
    fig_type = 'speaker_frequency_bar'
    span_granularities = ['word', 'line', 'scene']
    seasons = ti.xcom_pull(task_ids='fetch_seasons')
    for span_gran in span_granularities:
        for season in seasons:
            ap.publish_animation(show_key, fig_type, span_granularity=span_gran, season=season)


with DAG('publish_animations', start_date=datetime(2024, 10, 1),
         schedule_interval=None, catchup=False) as dag:
    '''
    Publish series- and season-level speaker_frequency_bar animations across span granularities
    '''

    publish_series_speaker_frequency_bars = PythonOperator(
        task_id='publish_series_speaker_frequency_bars',
        python_callable=_publish_series_speaker_frequency_bars
    )

    fetch_seasons = SimpleHttpOperator(
        task_id='fetch_seasons',
        http_conn_id='tp_api',
        endpoint='esr/list_seasons/TNG',
        method='GET',
        response_filter=lambda response: [season for season in response.json()['seasons']], 
        log_response=True
    )

    publish_season_speaker_frequency_bars = PythonOperator(
        task_id='publish_season_speaker_frequency_bars',
        python_callable=_publish_season_speaker_frequency_bars
    )
    
    publish_series_speaker_frequency_bars >> fetch_seasons >> publish_season_speaker_frequency_bars
