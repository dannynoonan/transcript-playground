from fastapi import APIRouter, status, HTTPException
from operator import itemgetter
import pandas as pd

from app.app_metadata import BERTOPIC_DATA_DIR
from app.auth import user_dependency
import app.data_service.field_flattener as fflat
import app.data_service.topicfidf_calculator as tfcalc
import app.database.dao as dao
import app.es.es_ingest_transformer as esit
import app.es.es_response_transformer as esrt
import app.es.es_query_builder as esqb
import app.es.es_read_router as esr
import app.es.es_write_router as esw
from app.nlp.nlp_metadata import ACTIVE_VENDOR_VERSIONS
from app.show_metadata import ShowKey, SPEAKERS_TO_IGNORE


esbw_app = APIRouter(tags=['ES Bulk Writer'])


##################### Legacy batch es writes, ported over to ./scripts (but still referenced by airflow dags) #######################

@esbw_app.get("/esw/index_all_episodes/{show_key}")
async def index_all_episodes(show_key: ShowKey, user: user_dependency, 
                             overwrite_all: bool = False):
    '''
    Bulk run of `/esw/index_episode` for all episodes of a given show
    NOTE migrated to ./scripts/index_episodes.py
    '''
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    
    episodes = []
    try:
        episodes = await dao.fetch_episodes(show_key.value)
    except Exception as e:
        return {"Error": f"Failure to fetch Episodes having show_key={show_key}: {e}"}
    if not episodes:
        return {"Error": f"No Episodes found having show_key={show_key}. You may need to run /load_episode_listing first."}
    if not overwrite_all:
        return {"No-op": f"/index_all_episodes was invoked on {len(episodes)} episodes, but `overwrite_all` flag was not set to True so no action was taken"}
    
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
        "episode_indexing_attempts": attempts, 
        "successful": len(successful_episode_keys),
        "successful_episode_keys": successful_episode_keys, 
        "failed": len(failed_episode_keys),
        "failed_episode_keys": failed_episode_keys, 
    }


@esbw_app.get("/esw/populate_all_episode_embeddings/{show_key}/{model_vendor}/{model_version}")
def populate_all_episode_embeddings(show_key: ShowKey, model_vendor: str, model_version: str, user: user_dependency):
    '''
    Bulk run of `/esw/populate_episode_embeddings` for all episodes of a given show
    NOTE migrated to ./scripts/populate_episode_embeddings.py
    '''
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    
    doc_ids = esr.fetch_doc_ids(ShowKey(show_key))
    episode_doc_ids = doc_ids['doc_ids']
    processed_episode_keys = []
    failed_episode_keys = []
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        try:
            esw.populate_episode_embeddings(ShowKey(show_key), episode_key, model_vendor, model_version)
            processed_episode_keys.append(episode_key)
        except Exception:
            failed_episode_keys.append(episode_key)
    return {"processed_episode_keys": processed_episode_keys, "failed_episode_keys": failed_episode_keys}


@esbw_app.get("/esw/populate_all_episode_relations/{show_key}/{model_vendor}/{model_version}")
def populate_all_episode_relations(show_key: ShowKey, model_vendor: str, model_version: str, user: user_dependency, 
                                   limit: int = 30):
    '''
    For each episode, query ElasticSearch for most similar episodes vis-a-vis a given model:vendor, then write the top X episode|score pairs to corresponding relations field
    NOTE migrated to ./scripts/populate_episode_relations.py
    '''
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    
    if (model_vendor, model_version) not in ACTIVE_VENDOR_VERSIONS and (model_vendor, model_version) != ('es','mlt'):
        return {"error": f'invalid model_vendor:model_version combo {model_vendor}:{model_version}'}
    
    doc_ids = esr.fetch_doc_ids(ShowKey(show_key))
    episode_doc_ids = doc_ids['doc_ids']
    
    episodes_to_relations = {}
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        if (model_vendor, model_version) == ('es','mlt'):
            similar_episodes = esr.more_like_this(ShowKey(show_key), episode_key)
        else:
            similar_episodes = esr.episode_mlt_vector_search(ShowKey(show_key), episode_key, model_vendor=model_vendor, model_version=model_version)
        # only keep the episode keys and corresponding scores 
        # sim_eps = [f"{sim_ep['episode_key']}|{sim_ep['score']}" for sim_ep in similar_episodes['matches']]
        episodes_to_relations[doc_id] = similar_episodes
    
    episodes_to_relations = esqb.populate_episode_relations(show_key.value, model_vendor, model_version, episodes_to_relations, limit=limit)

    return {"episodes_to_relations": episodes_to_relations}


@esbw_app.get("/esw/populate_topic_grouping_embeddings/{topic_grouping}/{model_vendor}/{model_version}")
def populate_topic_grouping_embeddings(topic_grouping: str, model_vendor: str, model_version: str, user: user_dependency):
    '''
    Generate vector embedding for all topics in topic_grouping using pre-trained Word2Vec and Transformer models
    NOTE migrated to ./scripts/populate_topic_embeddings.py
    '''
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    
    topic_grouping_response = esr.fetch_topic_grouping(topic_grouping)
    topic_keys = [t['topic_key'] for t in topic_grouping_response['topics']]
    attempted_count = 0
    successful_topics = []
    failed_topics = []
    failure_messages = []
    for topic_key in topic_keys:
        attempted_count += 1
        topic_embeddings_response = esw.populate_topic_embeddings(topic_grouping, topic_key, model_vendor, model_version)
        if 'topic' in topic_embeddings_response:
            successful_topics.append(topic_key)
        else:
            failed_topics.append(topic_key)
            if 'error' in topic_embeddings_response:
                failure_messages.append(topic_embeddings_response['error'])
        
    return {'attempted_count': attempted_count, 'successful_topics': successful_topics, 'failed_topics': failed_topics, 'failure_messages': failure_messages}


@esbw_app.get("/esw/index_all_speakers/{show_key}")
def index_all_speakers(show_key: ShowKey, user: user_dependency):
    '''
    Bulk run of `/esw/index_speaker` for all valid speakers with lines in a given show
    NOTE migrated to ./scripts/index_speakers.py
    '''
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    
    response = esr.agg_episodes_by_speaker(show_key)
    speaker_episode_counts = response['episodes_by_speaker']
    valid_speakers = [s for s,_ in speaker_episode_counts.items() if '+' not in s and s not in SPEAKERS_TO_IGNORE]
    attempt_count = 0
    successful = []
    failed = []
    for speaker in valid_speakers:
        attempt_count += 1
        try:
            response = esw.index_speaker(show_key, speaker)
            if "speaker" in response:
                print(f"Successfully indexed speaker={speaker}")
                successful.append(speaker)
            else:
                print(f"Failed to index speaker={speaker}: {response['Error']}")
                failed.append(speaker)
        except Exception as e:
            print(f"Failed to index speaker={speaker}: {e}")
            failed.append(speaker)

    return {"attempt_count": attempt_count, "successful": successful, "failed": failed}


@esbw_app.get("/esw/populate_all_speaker_embeddings/{show_key}/{model_vendor}/{model_version}")
def populate_all_speaker_embeddings(show_key: ShowKey, model_vendor: str, model_version: str, user: user_dependency):
    '''
    Generate vector embedding for all indexed speakers for a show using pre-trained Word2Vec and Transformer models
    NOTE migrated to ./scripts/populate_speaker_embeddings.py
    '''
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    
    s = esqb.fetch_indexed_speakers(show_key.value, return_fields=['speaker'])
    matches = esrt.return_speakers(s)
    if not matches:
        return {"error": f"Failed to fetch_indexed_speakers for show_key={show_key}"}
    
    speakers = [m['speaker'] for m in matches]
    request_count = 0
    success_count = 0
    skipped_count = 0
    failure_count = 0
    super_fails = []
    speaker_responses = {}
    for speaker in speakers:
        try:
            response = esw.populate_speaker_embeddings(show_key, speaker, model_vendor, model_version)
            speaker_responses[speaker] = response
            request_count += response['attempted_count']
            success_count += len(response['successful'])
            skipped_count += len(response['skipped'])
            failure_count += len(response['failed'])
        except Exception as e:
            print(f"Failed to populate_speaker_embeddings for speaker={speaker}: {e}")
            super_fails.append(speaker)

    return {"request_count": request_count, "success_count": success_count, "skipped_count": skipped_count, "failure_count": failure_count,
            "super_fails": super_fails, "speaker_responses": speaker_responses}


@esbw_app.get("/esw/populate_all_episode_topics/{show_key}/{topic_grouping}/{model_vendor}/{model_version}")
def populate_all_episode_topics(show_key: ShowKey, topic_grouping: str, model_vendor: str, model_version: str, user: user_dependency):
    '''
    For specified topic_grouping, generate and store topic mappings for all series episodes
    NOTE migrated to ./scripts/populate_episode_topics.py
    '''
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    
    doc_ids = esr.fetch_doc_ids(show_key)
    episode_doc_ids = doc_ids['doc_ids']
    processed_episode_keys = []
    failed_episode_keys = []
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        try:
            esw.populate_episode_topics(show_key, episode_key, topic_grouping, model_vendor, model_version)
            processed_episode_keys.append(episode_key)
        except Exception:
            failed_episode_keys.append(episode_key)

    return {"processed_episode_keys": processed_episode_keys, "failed_episode_keys": failed_episode_keys}


@esbw_app.get("/esw/populate_episode_topic_tfidf_scores/{show_key}/{topic_grouping}/{model_vendor}/{model_version}")
def populate_episode_topic_tfidf_scores(show_key: ShowKey, topic_grouping: str, model_vendor: str, model_version: str, user: user_dependency):
    '''
    For specified topic_grouping, calculate 'tfidf'-like scores for all episode_topics and store in `tfidf_score` field
    NOTE migrated to ./scripts/populate_episode_topics.py
    '''
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    
    ekey_tkey_scores, topic_idfs = tfcalc.calculate_topic_freq_idf(show_key, topic_grouping, model_vendor, model_version)

    successful_episode_keys = []
    # for each episode, generate and store "tf-idf" values per topic
    for e_key, t_keys_to_scores in ekey_tkey_scores.items():

        print(f'processing episode_topics for {e_key}')
        episode_topics = []
        for t_key, score in t_keys_to_scores.items():
            episode_topic = tfcalc.set_episode_topic_tfidf(show_key, t_key, e_key, score, topic_idfs, topic_grouping, model_vendor, model_version)
            episode_topics.append(episode_topic)
            
        # save simplified subset of episode_topics to es_episode.topics_X_tfidf
        # TODO topics_universal should either be topics_universal_{model_name} or a dict keying off of model_name instead of a list of topics
        # For now I'm only indexing topics generated with openai:3small embeddings 
        if topic_grouping in ['universalGenres'] and model_vendor == 'openai' and model_version == '3small':
            print(f'storing tfidf topics at episode level for {e_key}')
            tfidf_sorted_episode_topics = sorted(episode_topics, key=itemgetter('tfidf_score'), reverse=True)
            simple_episode_topics = fflat.flatten_es_topics(tfidf_sorted_episode_topics)
            esw.populate_episode_tfidf_topics(show_key, e_key, topic_grouping, simple_episode_topics)
            successful_episode_keys.append(e_key)
        else:
            successful_episode_keys.append(e_key)

    return {"attempted": len(ekey_tkey_scores), "successful": len(successful_episode_keys), "successful_episode_keys": successful_episode_keys}


@esbw_app.get("/esw/populate_all_speaker_topics/{show_key}/{topic_grouping}/{model_vendor}/{model_version}")
def populate_all_speaker_topics(show_key: ShowKey, topic_grouping: str, model_vendor: str, model_version: str, user: user_dependency):
    '''
    Map speakers to topics (using knn vector cosine similarity) for all indexed speakers for a show 
    NOTE migrated to ./scripts/populate_speaker_topics.py
    '''
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    
    s = esqb.fetch_indexed_speakers(show_key.value, return_fields=['speaker'])
    matches = esrt.return_speakers(s)
    if not matches:
        return {"error": f"Failed to fetch_indexed_speakers for show_key={show_key}"}
    
    speakers = [m['speaker'] for m in matches]
    attempt_count = 0
    success_count = 0
    successful_speakers = []
    failure_count = 0
    failed_speakers = []
    for speaker in speakers:
        attempt_count += 1
        try:
            response = esw.populate_speaker_topics(show_key, speaker, topic_grouping, model_vendor, model_version)
            if "error" in response:
                print(f"Failed to populate_speaker_topics for speaker={speaker}: {response['error']}")
                failed_speakers.append(speaker)
                failure_count += 1
            else:
                successful_speakers.append(speaker)
                success_count += 1
        except Exception as e:
            print(f"Failed to populate_speaker_topics for speaker={speaker}: {e}")
            failed_speakers.append(speaker)
            failure_count += 1

    return {"attempt_count": attempt_count, "success_count": success_count, "failure_count": failure_count,
            "successful_speakers": successful_speakers, "failed_speakers": failed_speakers}


@esbw_app.get("/esw/populate_all_episode_narratives/{show_key}/")
def populate_all_episode_narratives(show_key: ShowKey, user: user_dependency):
    '''
    Generate and populate all narrative sequences for a show
    NOTE migrated to ./scripts/populate_episode_narratives.py
    '''
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    
    successful_keys = []
    failed_keys = []

    doc_ids_response = esr.fetch_doc_ids(ShowKey(show_key))
    for doc_id in doc_ids_response['doc_ids']:
        episode_key = doc_id.split('_')[1]
        narrative_sequences_response = esw.populate_episode_narratives(show_key, episode_key)
        if 'narrative_sequences' not in narrative_sequences_response:
            print(f'Failed to populate_episode_narratives for show_key={show_key} episode_key={episode_key}')
            failed_keys.append(episode_key)
        else:
            successful_keys.append(episode_key)

    return {"successful_keys": successful_keys, "failed_keys": failed_keys}


@esbw_app.get("/esw/populate_bertopic_model_clusters/{show_key}/")
def populate_bertopic_model_clusters(show_key: ShowKey, user: user_dependency):
    '''
    Load each bertopic_model's csv into dataframe, upsert referenced episode_narratives with mapping back to bertopic_model
    NOTE migrated to ./scripts/populate_bertopic_clusters.py
    '''
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed')
    
    # load bertopic_data files 
    bertopic_model_list_response = esr.list_bertopic_models(show_key)
    bertopic_model_ids = bertopic_model_list_response['bertopic_model_ids']
    # NOTE umap_metric was supported previously, but the way I'm setting es_episode_narrative.cluster_memberships below precludes restricting by umap_metric 

    # initialize dict of narrative-speaker-groups per episode
    epnarr_spkrgrps_to_model_clusters = {}
    simple_episodes_response = esr.fetch_simple_episodes(show_key)
    if 'episodes' not in simple_episodes_response:
        print(f'Failure to /populate_bertopic_model_clusters for show_key={show_key}: /fetch_simple_episodes returned no episodes')
        return None
    for episode in simple_episodes_response['episodes']:
        e_key = episode['episode_key']
        narrative_sequences_response = esr.fetch_narrative_sequences(show_key, e_key)
        if 'narrative_sequences' not in narrative_sequences_response:
            print(f'Unable to /populate_bertopic_model_clusters for e_key={e_key} show_key={show_key}: /fetch_narrative_sequences returned no narrative_sequences. Skipping episode.')
            continue
        ep_narrs = narrative_sequences_response['narrative_sequences']
        epnarr_spkrgrps_to_model_clusters[e_key] = {narr['speaker_group']:[] for narr in ep_narrs}
    
    # populate episode-narrative-speaker-groups with any model_clusters of which they are a member
    for bertopic_model_id in bertopic_model_ids:
        df = pd.read_csv(f'{BERTOPIC_DATA_DIR}/{show_key.value}/{bertopic_model_id}.csv', sep='\t')
        # model_id = bertopic_model_id.removesuffix('.csv')
        for _, row in df.iterrows():
            e_key = str(row['episode_key'])
            spkr_grp = row['speaker_group']
            model_cluster = {}
            model_cluster['model_id'] = bertopic_model_id
            model_cluster['model_cluster_id'] = f"{bertopic_model_id}__{e_key}_{row['cluster_id']}__{spkr_grp}"
            model_cluster['probability'] = row['Probability']
            model_cluster['prob_x_wc'] = row['prob_x_wc']
            model_cluster['cluster_title'] = row['cluster_title']
            # convert stringified cluster_keywords back into list
            cluster_keywords = row['cluster_keywords'].split("', '")
            if cluster_keywords:
                cluster_keywords[0] = cluster_keywords[0].removeprefix("['")
                cluster_keywords[len(cluster_keywords)-1] = cluster_keywords[len(cluster_keywords)-1].removesuffix("']")
            model_cluster['cluster_keywords'] = cluster_keywords
            epnarr_spkrgrps_to_model_clusters[e_key][spkr_grp].append(model_cluster)

    # update all cluster_memberships for any given episode-narrative at once, as opposed to piece-meal incrementally (since we don't have any criteria for deleting old mappings)
    attempt_count = 0
    success_count = 0
    failure_count = 0
    for e_key, spkr_grps_to_clusters in epnarr_spkrgrps_to_model_clusters.items():
        for spkr_grp, model_clusters in spkr_grps_to_clusters.items():
            attempt_count += 1
            es_episode_narrative = esqb.fetch_episode_narrative(show_key, e_key, spkr_grp)
            if not es_episode_narrative:
                print(f'Failure to update cluster_memberships for episode narrative: no EsEpisodeNarrativeSequence found matching show_key={show_key} e_key={e_key} spkr_grp={spkr_grp}. Skipping to next episode narrative.')
                failure_count += 1
                continue
            es_episode_narrative.cluster_memberships = model_clusters
            try:
                esqb.save_episode_narrative(es_episode_narrative)
                success_count += 1
            except Exception as e:
                print(f'Failure to update cluster_memberships for episode narrative at show_key={show_key} e_key={e_key} spkr_grp={spkr_grp}: {e}')
                failure_count += 1

    return {"attempt_count": attempt_count, "success_count": success_count, "failure_count": failure_count}


# @esbw_app.get("/esw/generate_all_episode_polarity_sentiments/{show_key}")
# def generate_all_episode_polarity_sentiments(show_key: ShowKey, scene_level: bool = False, scene_event_level: bool = False):
#     '''
#     Generate and  populate nltk polarity sentiment for all episodes in series
#     '''
#     episodes_with_sentiment = []
#     # successful = []
#     # failed = []

#     simple_episodes_response = esr.fetch_simple_episodes(show_key)
#     simple_episodes = simple_episodes_response['episodes']
#     for se in simple_episodes:
#         episode_sentiment_response = populate_episode_polarity_sentiment(show_key, se['episode_key'], scene_level=scene_level, scene_event_level=scene_event_level)
#         episodes_with_sentiment.append(episode_sentiment_response)

#     return {f"episodes_with_sentiment": episodes_with_sentiment}
