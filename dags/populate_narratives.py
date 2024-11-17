from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator


with DAG('populate_narratives', start_date=datetime(2024, 10, 1),
         schedule_interval=None, catchup=False) as dag:
    '''
    TODO
    '''

    populate_episode_narratives = SimpleHttpOperator(
        task_id='populate_episode_narratives',
        http_conn_id='tp_api',
        endpoint='esw/populate_all_episode_narratives/TNG',
        method='GET',
        response_filter=lambda response: response.json()['successful_keys'], 
        log_response=True
    )

    populate_bertopic_clusters = SimpleHttpOperator(
        task_id='populate_bertopic_clusters',
        http_conn_id='tp_api',
        endpoint='esw/populate_bertopic_model_clusters/TNG',
        method='GET',
        response_filter=lambda response: response.json()['success_count'], 
        log_response=True
    )
    
    populate_episode_narratives >> populate_bertopic_clusters
