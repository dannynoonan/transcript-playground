from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

from app.database import dao
import app.es.es_ingest_transformer as esit
import app.es.es_query_builder as esqb
from app.show_metadata import ShowKey


def _fetch_psql_episode(ti):
    show_key = 'TNG'
    episode_key = '218'
    print(f"In _fetch_psql_episode show_key={show_key} episode_key={episode_key}")
    episode = None
    try:
        episode = dao.fetch_episode(ShowKey(show_key), episode_key, fetch_related=['scenes', 'events'])
        print(f"Success: Episode having show_key={show_key} episode_key={episode_key} fetched from psql")
    except Exception as e:
        print(f"Error: Failure to fetch Episode having show_key={show_key} episode_key={episode_key} (have you run /load_episode_listing?): {e}")

    if not episode:
        print(f"Error: No Episode found having show_key={show_key} episode_key={episode_key}. You may need to run /load_episode_listing first.")

    ti.xcom_push(key='episode', value=episode)


def _write_psql_episode_to_es(ti):
    psql_episode = ti.xcom_pull(key='episode', task_ids='fetch_psql_episode')
    print(f"In _write_psql_episode_to_es psql_episode={psql_episode}")

    # transform to es-writable object and write to es
    try:
        es_episode = esit.to_es_episode(psql_episode)
        esqb.save_es_episode(es_episode)
        print(f"Success: Episode loaded into es")
    except Exception as e:
        print(f"Error: Failure to transform episode to es-writable version: {e}")


# def _fetch_psql_episode(ti):
#     ti.xcom_push(key='my_key', value=42)
 
# def _write_psql_episode_to_es(ti):
#     print(ti.xcom_pull(key='my_key', task_ids='fetch_psql_episode'))


with DAG('index_episode', start_date=datetime(2024, 10, 1),
         schedule_interval='@daily', catchup=False) as dag:
    
    '''
    Fetch episode from transcript_db, transform to es object, write to es index
    '''

    fetch_psql_episode = PythonOperator(
        task_id='fetch_psql_episode',
        python_callable=_fetch_psql_episode
    )
    
    write_psql_episode_to_es = PythonOperator(
        task_id='write_psql_episode_to_es',
        python_callable=_write_psql_episode_to_es
    )
    
    fetch_psql_episode >> write_psql_episode_to_es
