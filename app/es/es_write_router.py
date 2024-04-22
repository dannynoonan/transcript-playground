from fastapi import APIRouter
import os
import numpy as np
import pandas as pd

import app.database.dao as dao
import app.es.es_ingest_transformer as esit
from app.es.es_metadata import ACTIVE_VENDOR_VERSIONS
from app.es.es_model import EsEpisodeTranscript, EsTopic
import app.es.es_query_builder as esqb
import app.es.es_read_router as esr
import app.nlp.embeddings_factory as ef
from app.show_metadata import ShowKey


esw_app = APIRouter()



@esw_app.get("/esw/init_es", tags=['ES Writer'])
def init_es(index_name: str = None):
    '''
    Run this to explicitly define the mapping anytime the `transcripts` index is blown away and re-created. Not doing so will result in the wrong
    data types being auto-assigned to several fields in the schema mapping, and will break query (read) functionality down the line.
    '''
    valid_indexes = ['transcripts', 'characters', 'topics']
    if index_name:
        if index_name not in valid_indexes:
            return {"error": "Failed to initialize index_name={index_name}, valid_indexes={valid_indexes}"}
        if index_name == 'transcripts':
            esqb.init_transcripts_index()
        elif index_name == 'characters':
            esqb.init_characters_index()
        elif index_name == 'topics':
            esqb.init_topics_index()
        initialized_indexes = [index_name]
    else:
        esqb.init_transcripts_index()
        esqb.init_characters_index()
        esqb.init_topics_index()
        initialized_indexes = valid_indexes

    return {"initialized_indexes": initialized_indexes}


@esw_app.get("/esw/index_episode/{show_key}/{episode_key}", tags=['ES Writer'])
async def index_transcript(show_key: ShowKey, episode_key: str):
    '''
    Fetch `Episode` entity from Postgres `transcript_db`, transform Tortoise object to ElasticSearch object, and write it to ElasticSearch index.
    '''
    # fetch episode, throw errors if not found
    episode = None
    try:
        episode = await dao.fetch_episode(show_key.value, episode_key, fetch_related=['scenes', 'events'])
    except Exception as e:
        return {"Error": f"Failure to fetch Episode having show_key={show_key} external_key={episode_key} (have run /load_episode_listing?): {e}"}
    if not episode:
        return {"Error": f"No Episode found having show_key={show_key} external_key={episode_key}. You may need to run /load_episode_listing first."}
    
    # transform to es-writable object and write to es
    try:
        es_episode = esit.to_es_episode(episode)

        # NOTE this is how I generated test data, and it was enough of a pain I don't want to delete it
        # import json
        # es_episode_dict = es_episode.to_dict()
        # es_episode_json = json.dumps(es_episode_dict, default=str, indent=4)
        # f = open(f"test_data/es/es_episode_{show_key}_{episode_key}.json", "w")
        # f.write(es_episode_json)
        # f.close()
            
        esqb.save_es_episode(es_episode)
    except Exception as e:
        return {"Error": f"Failure to transform Episode {show_key}:{episode_key} to es-writable version: {e}"}

    return {"Success": f"Episode {show_key}_{episode_key} written to es index"}


@esw_app.get("/esw/index_all_episodes/{show_key}", tags=['ES Writer'])
async def index_all_transcripts(show_key: ShowKey, overwrite_all: bool = False):
    '''
    Bulk run of `/esw/index_episode` for all episodes of a given show
    '''
    episodes = []
    try:
        episodes = await dao.fetch_episodes(show_key.value)
    except Exception as e:
        return {"Error": f"Failure to fetch Episodes having show_key={show_key}: {e}"}
    if not episodes:
        return {"Error": f"No Episodes found having show_key={show_key}. You may need to run /load_episode_listing first."}
    if not overwrite_all:
        return {"No-op": f"/index_transcripts was invoked on {len(episodes)} episodes, but `overwrite_all` flag was not set to True so no action was taken"}
    
    # fetch and insert transcripts for all episodes
    attempts = 0
    successful_episode_keys = []
    failed_episode_keys = []
    for episode in episodes:
        attempts += 1

        # fetch nested scene and scene_event data
        await episode.fetch_related('scenes')
        # if not episode.scenes:
        #     print(f"No Scene data found for episode {show_key}_{episode.external_key}. You may need to run /load_transcript first.")
        #     failed_episode_keys.append(episode.external_key)
        #     continue
        for scene in episode.scenes:
            await scene.fetch_related('events')

        # transform to es-writable object and write to es
        try:
            es_episode = esit.to_es_episode(episode)
            esqb.save_es_episode(es_episode)
            successful_episode_keys.append(episode.external_key)
        except Exception as e:
            failed_episode_keys.append(episode.external_key)
            print(f"Failure to transform Episode {show_key}_{episode.external_key} to es-writable version or write it to es: {e}")

    return {
        "index loading attempts": attempts, 
        "successful": len(successful_episode_keys),
        "successful episode keys": successful_episode_keys, 
        "failed": len(failed_episode_keys),
        "failed episode keys": failed_episode_keys, 
    }


@esw_app.get("/esw/populate_focal_speakers/{show_key}", tags=['ES Writer'])
async def populate_focal_speakers(show_key: ShowKey, episode_key: str = None):
    '''
    For each episode, query ElasticSearch to count the number of lines spoken per character, then write the top 3 characters back to their own ElasticSearch field
    '''
    episodes_to_focal_speakers = await esqb.populate_focal_speakers(show_key.value, episode_key)
    return {"episodes_to_focal_speakers": episodes_to_focal_speakers}


@esw_app.get("/esw/populate_focal_locations/{show_key}", tags=['ES Writer'])
async def populate_focal_locations(show_key: ShowKey, episode_key: str = None):
    '''
    For each episode, query ElasticSearch to count the number of scenes per location, then write the top 3 locations back to their own ElasticSearch field
    '''
    episodes_to_focal_locations = await esqb.populate_focal_locations(show_key.value, episode_key)
    return {"episodes_to_focal_locations": episodes_to_focal_locations}


@esw_app.get("/esw/populate_relations/{show_key}/{episode_key}/{model_vendor}/{model_version}", tags=['ES Writer'])
async def populate_relations(show_key: ShowKey, episode_key: str, model_vendor: str, model_version: str, limit: int = 30):
    '''
    Query ElasticSearch for most similar episodes vis-a-vis a given model:vendor, then write the top X episode|score pairs to corresponding relations field
    '''
    if (model_vendor, model_version) not in ACTIVE_VENDOR_VERSIONS and (model_vendor, model_version) != ('es','mlt'):
        return {"error": f'invalid model_vendor:model_version combo {model_vendor}:{model_version}'}
 
    if (model_vendor, model_version) == ('es','mlt'):
        similar_episodes = await esr.more_like_this(ShowKey(show_key), episode_key)
    else:
        similar_episodes = esr.mlt_vector_search(ShowKey(show_key), episode_key, model_vendor=model_vendor, model_version=model_version)
    # only keep the episode keys and corresponding scores 
    # sim_eps = [f"{sim_ep['episode_key']}|{sim_ep['score']}" for sim_ep in similar_episodes['matches']]
    # sim_eps = [(sim_ep['episode_key'], sim_ep['score']) for sim_ep in similar_episodes['matches']]
    # sim_eps = {sim_ep['episode_key']:sim_ep['score'] for sim_ep in similar_episodes['matches']}

    episode_relations = {}
    doc_id = f'{show_key}_{episode_key}'
    episode_relations[doc_id] = similar_episodes
    
    episode_relations = await esqb.populate_relations(show_key.value, model_vendor, model_version, episode_relations, limit=limit)

    return {"episode_relations": episode_relations}


@esw_app.get("/esw/populate_all_relations/{show_key}/{model_vendor}/{model_version}", tags=['ES Writer'])
async def populate_relations(show_key: ShowKey, model_vendor: str, model_version: str, limit: int = 30, episode_key: str = None):
    '''
    For each episode, query ElasticSearch for most similar episodes vis-a-vis a given model:vendor, then write the top X episode|score pairs to corresponding relations field
    '''
    if (model_vendor, model_version) not in ACTIVE_VENDOR_VERSIONS and (model_vendor, model_version) != ('es','mlt'):
        return {"error": f'invalid model_vendor:model_version combo {model_vendor}:{model_version}'}
    
    doc_ids = esr.fetch_doc_ids(ShowKey(show_key))
    episode_doc_ids = doc_ids['doc_ids']
    
    episodes_to_relations = {}
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        if (model_vendor, model_version) == ('es','mlt'):
            similar_episodes = await esr.more_like_this(ShowKey(show_key), episode_key)
        else:
            similar_episodes = esr.mlt_vector_search(ShowKey(show_key), episode_key, model_vendor=model_vendor, model_version=model_version)
        # only keep the episode keys and corresponding scores 
        # sim_eps = [f"{sim_ep['episode_key']}|{sim_ep['score']}" for sim_ep in similar_episodes['matches']]
        episodes_to_relations[doc_id] = similar_episodes
    
    episodes_to_relations = await esqb.populate_relations(show_key.value, model_vendor, model_version, episodes_to_relations, limit=limit)

    return {"episodes_to_relations": episodes_to_relations}


@esw_app.get("/esw/build_embeddings_model/{show_key}", tags=['ES Writer'])
def build_embeddings_model(show_key: ShowKey):
    '''
    Experimental endpoint: goes thru the motions of building a language model using Word2Vec, but limits training data to a single show's text corpus, resulting in a (uselessly) tiny model
    '''
    model_info = ef.build_embeddings_model(show_key.value)
    return {"model_info": model_info}


@esw_app.get("/esw/populate_embeddings/{show_key}/{episode_key}/{model_vendor}/{model_version}", tags=['ES Writer'])
def populate_embeddings(show_key: ShowKey, episode_key: str, model_vendor: str, model_version: str):
    '''
    Generate vector embedding for episode using pre-trained Word2Vec and Transformer models (enumerated in `nlp/nlp_metadata.py`)
    '''
    es_episode = EsEpisodeTranscript.get(id=f'{show_key.value}_{episode_key}')
    try:
        ef.generate_episode_embeddings(show_key.value, es_episode, model_vendor, model_version)
        esqb.save_es_episode(es_episode)
        return {"es_episode": es_episode}
    except Exception as e:
        return {f"Failed to populate {model_vendor}:{model_version} embeddings for episode {show_key.value}_{episode_key}": e}


@esw_app.get("/esw/populate_all_embeddings/{show_key}/{model_vendor}/{model_version}", tags=['ES Writer'])
def populate_all_embeddings(show_key: ShowKey, model_vendor: str, model_version: str):
    '''
    Bulk run of `/esw/populate_embeddings` for all episodes of a given show
    '''
    doc_ids = esr.fetch_doc_ids(ShowKey(show_key))
    episode_doc_ids = doc_ids['doc_ids']
    processed_episode_keys = []
    failed_episode_keys = []
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        try:
            populate_embeddings(ShowKey(show_key), episode_key, model_vendor, model_version)
            processed_episode_keys.append(episode_key)
        except Exception:
            failed_episode_keys.append(episode_key)
    return {"processed_episode_keys": processed_episode_keys, "failed_episode_keys": failed_episode_keys}


@esw_app.get("/esw/index_topic_grouping/{topic_grouping}", tags=['ES Writer'])
async def index_topic_grouping(topic_grouping: str, concat_hierarchy_text: bool = False):
    '''
    Load set of Topics from csv file into es `topics` index.
    '''
    file_path = f'./source/topics/{topic_grouping}.csv'
    if os.path.isfile(file_path):
        print(f'Loading topic_grouping dataframe from file_path={file_path}')
        topics_df = pd.read_csv(file_path)
        topics_df = topics_df.fillna('')
        # optionally prefix subcategory descriptions with their parent category descriptions before generating embeddings, for fuller description context
        if concat_hierarchy_text:
            distinct_cats = topics_df['category'].unique()
            for dc in distinct_cats:
                cat_desc_ser = topics_df[(topics_df['category'] == dc) & (topics_df['subcategory'] == '')]['description']
                cat_desc = cat_desc_ser.values[0] # NOTE feels like there should be a cleaner way to extract the category description
                topics_df.loc[(topics_df['category'] == dc) & (topics_df['subcategory'] != ''), 'description'] = cat_desc + '\n\n' + topics_df['description']
            # new_file_path = f'./source/topics/{topic_grouping}_concat.csv'
            # topics_df.to_csv(new_file_path, sep='\t')
    else:
        return {'Error': f'Unable to load topics for topic_grouping={topic_grouping}, no file found at file_path={file_path}'}
    
    es_topics = []
    for _, row in topics_df.iterrows():
        es_topic = EsTopic(topic_grouping=topic_grouping, topic_key=row['key'], category=row['category'], subcategory=row['subcategory'], description=row['description'])
        es_topic.breadcrumb = es_topic.category
        if es_topic.subcategory:
            es_topic.breadcrumb = f'{es_topic.breadcrumb} > {es_topic.subcategory}'
            es_topic.name = es_topic.subcategory
        else:
            es_topic.name = es_topic.category
        es_topics.append(es_topic)
    
    # write to es
    attempted_count = 0
    successful_topics = []
    failed_topics = []
    for es_topic in es_topics:
        attempted_count += 1
        try:
            esqb.save_es_topic(es_topic)
            successful_topics.append(es_topic.topic_key)
        except Exception as e:
            failed_topics.append(es_topic.topic_key)

    return {'attempted_count': attempted_count, 'successful_topics': successful_topics, 'failed_topics': failed_topics}


@esw_app.get("/esw/populate_topic_embeddings/{topic_grouping}/{topic_key}/{model_vendor}/{model_version}", tags=['ES Writer'])
def populate_topic_embeddings(topic_grouping: str, topic_key: str, model_vendor: str, model_version: str):
    '''
    Generate vector embedding for topic using pre-trained Word2Vec and Transformer models
    '''
    doc_id = f'{topic_grouping}_{topic_key}'
    es_topic = EsTopic.get(id=doc_id)
    try:
        ef.generate_topic_embeddings(es_topic, model_vendor, model_version)
        esqb.save_es_episode(es_topic)
        return {"topic": es_topic._d_}
    except Exception as e:
        return {f"error": "Failed to populate {model_vendor}:{model_version} embeddings for topic {topic_grouping}:{topic_key}, {e}"}
    

@esw_app.get("/esw/populate_topic_grouping_embeddings/{topic_grouping}/{model_vendor}/{model_version}", tags=['ES Writer'])
def populate_topic_grouping_embeddings(topic_grouping: str, model_vendor: str, model_version: str):
    '''
    Generate vector embedding for all topics in topic_grouping using pre-trained Word2Vec and Transformer models
    '''
    topic_grouping_response = esr.fetch_topic_grouping(topic_grouping)
    topic_keys = [t['topic_key'] for t in topic_grouping_response['topics']]
    attempted_count = 0
    successful_topics = []
    failed_topics = []
    failure_messages = []
    for topic_key in topic_keys:
        attempted_count += 1
        topic_embeddings_response = populate_topic_embeddings(topic_grouping, topic_key, model_vendor, model_version)
        if 'topic' in topic_embeddings_response:
            successful_topics.append(topic_key)
        else:
            failed_topics.append(topic_key)
            if 'error' in topic_embeddings_response:
                failure_messages.append(topic_embeddings_response['error'])
        
    return {'attempted_count': attempted_count, 'successful_topics': successful_topics, 'failed_topics': failed_topics, 'failure_messages': failure_messages}
