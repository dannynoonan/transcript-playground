from fastapi import APIRouter
from operator import itemgetter
import os
import pandas as pd

import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.nlp.embeddings_factory as ef
from app.nlp.nlp_metadata import WORD2VEC_VENDOR_VERSIONS as W2V_MODELS, TRANSFORMER_VENDOR_VERSIONS as TRF_MODELS, BERTOPIC_MODELS_DIR
import app.nlp.query_preprocessor as qp
from app.show_metadata import ShowKey, show_metadata, EPISODE_TOPIC_GROUPINGS


esr_app = APIRouter()



###################### SIMPLE FETCH ###########################

@esr_app.get("/esr/episode/{show_key}/{episode_key}", tags=['ES Reader'])
def fetch_episode(show_key: ShowKey, episode_key: str, all_fields: bool = False):
    '''
    Fetch individual episode 
    '''
    s = esqb.fetch_episode_by_key(show_key.value, episode_key, all_fields=all_fields)
    es_query = s.to_dict()
    match = esrt.return_episode_by_key(s)
    return {"es_episode": match, 'es_query': es_query}


@esr_app.get("/esr/fetch_doc_ids/{show_key}", tags=['ES Reader'])
def fetch_doc_ids(show_key: ShowKey, season: str = None):
    '''
    Get all es source _ids for show 
    '''
    s = esqb.fetch_doc_ids(show_key.value, season=season)
    es_query = s.to_dict()
    matches = esrt.return_doc_ids(s)
    return {"doc_count": len(matches), "doc_ids": matches, "es_query": es_query}


@esr_app.get("/esr/list_seasons/{show_key}", tags=['ES Reader'])
def list_seasons(show_key: ShowKey):
    '''
    List all distincts seasons, sorted ascending
    '''
    s = esqb.agg_seasons(show_key.value)
    es_query = s.to_dict()
    seasons = esrt.return_seasons(s)
    seasons = sorted(seasons)
    return {"seasons": seasons, "es_query": es_query}


# NOTE I'm not sure what this was for, probably created during BERTopic experimentation
# @esr_app.get("/esr/fetch_flattened_episodes/{show_key}", tags=['ES Reader'])
# def fetch_flattened_episodes(show_key: ShowKey, season: str = None):
#     '''
#     Fetch episodes with full flattened text, but lacking scene or scene_event structure
#     '''
#     s = esqb.fetch_flattened_episodes(show_key.value, season=season)
#     es_query = s.to_dict()
#     episodes = esrt.return_simple_episodes(s)
#     return {"episodes": episodes, "es_query": es_query}


@esr_app.get("/esr/fetch_simple_episodes/{show_key}", tags=['ES Reader'])
def fetch_simple_episodes(show_key: ShowKey, season: str = None):
    '''
    Fetch simple (sceneless) episodes 
    '''
    s = esqb.fetch_simple_episodes(show_key.value, season=season)
    es_query = s.to_dict()
    episodes = esrt.return_simple_episodes(s)
    return {"episodes": episodes, "es_query": es_query}


@esr_app.get("/esr/list_simple_episodes_by_season/{show_key}", tags=['ES Reader'])
def list_simple_episodes_by_season(show_key: ShowKey):
    '''
    Fetch simple (sceneless) episodes sequenced and grouped by season
    '''
    s = esqb.fetch_simple_episodes(show_key.value)
    es_query = s.to_dict()
    episodes_by_season = esrt.return_episodes_by_season(s)
    return {"episodes_by_season": episodes_by_season, "es_query": es_query}


@esr_app.get("/esr/fetch_episode_narrative/{show_key}/{episode_key}/{speaker_group}", tags=['ES Reader'])
def fetch_episode_narrative(show_key: ShowKey, episode_key: str, speaker_group: str):
    '''
    Fetch individual pre-generated narrative sequence for a given episode + speaker_group
    '''
    episode_narrative = esqb.fetch_episode_narrative(show_key.value, episode_key, speaker_group)
    if not episode_narrative:
        return {"error": f"Failed to /fetch_episode_narrative for show_key=`{show_key.value}` episode_key=`{episode_key}` speaker_group=`{speaker_group}`"}
    episode_narrative = episode_narrative._d_

    return {"episode_narrative": episode_narrative}


@esr_app.get("/esr/fetch_narrative_sequences/{show_key}/{episode_key}", tags=['ES Reader'])
def fetch_narrative_sequences(show_key: ShowKey, episode_key: str):
    '''
    Fetch pre-generated narrative sequences for a given episode
    '''
    s = esqb.fetch_narrative_sequences(show_key.value, episode_key)
    es_query = s.to_dict()
    narrative_sequences = esrt.return_narrative_sequences(s)
    return {"narrative_sequences": narrative_sequences, "es_query": es_query}


@esr_app.get("/esr/fetch_flattened_scenes/{show_key}/{episode_key}", tags=['ES Reader'])
def fetch_flattened_scenes(show_key: ShowKey, episode_key: str, include_speakers: bool = False, include_context: bool = False, line_breaks: bool = False):
    '''
    Fetch denormalized scene text for a given episode
    '''
    s = esqb.fetch_episode_by_key(show_key.value, episode_key)
    es_query = s.to_dict()
    flattened_scenes = esrt.return_flattened_scenes(s, include_speakers=include_speakers, include_context=include_context, line_breaks=line_breaks)
    return {"flattened_scenes": flattened_scenes, "es_query": es_query}
 

@esr_app.get("/esr/fetch_all_episode_relations/{show_key}/{model_vendor}/{model_version}", tags=['ES Reader'])
def fetch_all_episode_relations(show_key: ShowKey, model_vendor: str, model_version: str):
    '''
    Fetch all (sceneless) episodes and their relations data for a given model vendor/version
    '''
    s = esqb.fetch_all_episode_relations(show_key.value, model_vendor, model_version)
    es_query = s.to_dict()
    episode_relations = esrt.return_all_episode_relations(s)
    return {"episode_relations": episode_relations, "es_query": es_query}


@esr_app.get("/esr/speaker/{show_key}/{speaker_name}", tags=['ES Reader'])
def fetch_speaker(show_key: ShowKey, speaker_name: str, include_seasons: bool = False, include_episodes: bool = False):
    '''
    Fetch speaker info, lines, and aggregate counts (optionally across seasons and episodes)
    '''
    speaker = esqb.fetch_speaker(show_key.value, speaker_name)
    if not speaker:
        return {"error": f"Failed to fetch speaker `{speaker_name}` for show_key=`{show_key.value}`"}
    speaker.seasons_to_episode_keys = speaker.seasons_to_episode_keys._d_
    speaker = speaker._d_

    es_queries = []
    if include_seasons:
        s = esqb.fetch_speaker_seasons(show_key.value, speaker=speaker_name)
        es_queries.append(s.to_dict())
        speaker_seasons = esrt.return_speaker_seasons(s)
        if speaker_seasons:
            speaker['seasons'] = speaker_seasons
    if include_episodes:
        s = esqb.fetch_speaker_episodes(show_key.value, speaker=speaker_name)
        es_queries.append(s.to_dict())
        speaker_episodes = esrt.return_speaker_episodes(s)
        if speaker_episodes:
            speaker['episodes'] = speaker_episodes

    return {"speaker": speaker, "es_queries": es_queries}


@esr_app.get("/esr/fetch_speakers_for_episode/{show_key}/{episode_key}", tags=['ES Reader'])
def fetch_speakers_for_episode(show_key: ShowKey, episode_key: str, extra_fields: str = None):
    '''
    Fetch speaker_episodes for a given episode 
    '''
    return_fields = ['speaker','scene_count','line_count','word_count','agg_score']
    if extra_fields:
        extra_fields = extra_fields.split(',')
        return_fields.extend(extra_fields)

    s = esqb.fetch_speaker_episodes(show_key.value, episode_key=episode_key, return_fields=return_fields)
    es_query = s.to_dict()
    speaker_episodes = esrt.return_speaker_episodes(s)

    return {"speaker_episodes": speaker_episodes, "es_query": es_query}


@esr_app.get("/esr/fetch_speakers_for_season/{show_key}/{season}", tags=['ES Reader'])
def fetch_speakers_for_season(show_key: ShowKey, season: str, extra_fields: str = None):
    '''
    Fetch speaker_seasons for a given season 
    '''
    return_fields = ['speaker','episode_count','scene_count','line_count','word_count','agg_score']
    if extra_fields:
        extra_fields = extra_fields.split(',')
        return_fields.extend(extra_fields)

    s = esqb.fetch_speaker_seasons(show_key.value, season=season, return_fields=return_fields)
    es_query = s.to_dict()
    speaker_seasons = esrt.return_speaker_seasons(s)

    return {"speaker_seasons": speaker_seasons, "es_query": es_query}


@esr_app.get("/esr/fetch_indexed_speakers/{show_key}", tags=['ES Reader'])
def fetch_indexed_speakers(show_key: ShowKey, speakers: str = None, season: int = None, extra_fields: str = None, min_episode_count: int = None):
    '''
    For speakers indexed in es, fetch info, lines, and aggregate counts
    '''
    return_fields = ['speaker', 'alt_names', 'actor_names', 'season_count', 'episode_count', 'scene_count', 'line_count', 'word_count', 'openai_ada002_word_count']
    speaker_list = []
    if speakers:
        speaker_list = speakers.split(',')
    if extra_fields:
        extra_fields = extra_fields.split(',')
        return_fields.extend(extra_fields)

    # NOTE I'm not sure the `season` parameter will last, it excludes speakers by season but still returns series-level agg counts
    s = esqb.fetch_indexed_speakers(show_key.value, speaker_list=speaker_list, season=season, return_fields=return_fields, min_episode_count=min_episode_count)
    es_query = s.to_dict()
    speakers = esrt.return_speakers(s)
    if not speakers:
        return {"error": f"Failed to fetch_indexed_speakers for show_key={show_key}"}

    return {"speakers": speakers, "es_query": es_query}


@esr_app.get("/esr/topic/{topic_grouping}/{topic_key}", tags=['ES Reader'])
def fetch_topic(topic_grouping: str, topic_key: str):
    '''
    Fetch individual topic 
    '''
    topic = esqb.fetch_topic(topic_grouping, topic_key)
    if not topic:
        return {"error": f"Failed to fetch topic for topic_grouping=`{topic_grouping}` topic_key=`{topic_key}`"}
    return {"topic": topic._d_}


@esr_app.get("/esr/fetch_topic_grouping/{topic_grouping}", tags=['ES Reader'])
def fetch_topic_grouping(topic_grouping: str, return_fields: str = None):
    '''
    Fetch all topics in a topic_grouping 
    '''
    if return_fields:
        return_fields = return_fields.split(',')
    s = esqb.fetch_topic_grouping(topic_grouping, return_fields=return_fields)
    es_query = s.to_dict()
    topics = esrt.return_topics(s)
    return {"topics": topics, "es_query": es_query}


@esr_app.get("/esr/fetch_episode_topics/{show_key}/{episode_key}/{topic_grouping}", tags=['ES Reader'])
def fetch_episode_topics(show_key: ShowKey, episode_key: str, topic_grouping: str, level: str = None, limit: int = None, sort_by: str = None):
    '''
    Fetch topics mapped to episode
    '''
    s = esqb.fetch_episode_topics(show_key.value, episode_key, topic_grouping, level=level, limit=limit, sort_by=sort_by)
    es_query = s.to_dict()
    episode_topics = esrt.return_topics(s)
    return {"episode_topics": episode_topics, "es_query": es_query}


@esr_app.get("/esr/fetch_speaker_topics/{speaker}/{show_key}/{topic_grouping}", tags=['ES Reader'])
def fetch_speaker_topics(speaker: str, show_key: ShowKey, topic_grouping: str, level: str = None, limit: int = None):
    '''
    Fetch topics mapped to speaker 
    '''
    s = esqb.fetch_speaker_topics(speaker, show_key.value, topic_grouping, level=level, limit=limit)
    es_query = s.to_dict()
    speaker_topics = esrt.return_topics(s)
    return {"speaker_topics": speaker_topics, "es_query": es_query}


@esr_app.get("/esr/fetch_speaker_season_topics/{show_key}/{topic_grouping}", tags=['ES Reader'])
def fetch_speaker_season_topics(show_key: ShowKey, topic_grouping: str, speaker: str = None, season: int = None, level: str = None, limit: int = None):
    '''
    Fetch topics mapped to speaker_season
    '''
    s = esqb.fetch_speaker_season_topics(show_key.value, topic_grouping, speaker=speaker, season=season, level=level, limit=limit)
    es_query = s.to_dict()
    # TODO feels like I should either do this in more places or not do it here
    if speaker:
        speaker_season_topics = esrt.return_topics_by_season(s)
    else:
        speaker_season_topics = esrt.return_topics_by_speaker(s)
    return {"speaker_season_topics": speaker_season_topics, "es_query": es_query}


@esr_app.get("/esr/fetch_speaker_episode_topics/{show_key}/{topic_grouping}", tags=['ES Reader'])
def fetch_speaker_episode_topics(show_key: ShowKey, topic_grouping: str, speaker: str = None, episode_key: str = None, season: int = None, 
                                 level: str = None, limit: int = None):
    '''
    Fetch topics mapped to speaker_episode
    '''
    s = esqb.fetch_speaker_episode_topics(show_key.value, topic_grouping, speaker=speaker, episode_key=episode_key, season=season, level=level, limit=limit)
    es_query = s.to_dict()
    speaker_episode_topics = esrt.return_topics_by_episode(s)
    return {"speaker_episode_topics": speaker_episode_topics, "es_query": es_query}


@esr_app.get("/esr/list_bertopic_models/{show_key}", tags=['ES Reader'])
def list_bertopic_models(show_key: str, umap_metric: str = None):
    '''
    List bertopic models for show using a directory scan
    NOTE this isn't an `es_read` but it might be in the future (and there's no other obvious place for it to live)
    '''
    bertopic_models_dir = f'{BERTOPIC_MODELS_DIR}/{show_key}'
    # bertopic_model_id_options = [f.removesuffix('.csv') for f in os.listdir(bertopic_data_dir) if os.path.isfile(os.path.join(bertopic_data_dir, f))]
    bertopic_model_ids = [m for m in os.listdir(bertopic_models_dir)]
    if umap_metric:
        bertopic_model_ids = [m for m in bertopic_model_ids if m.startswith(umap_metric)]
    bertopic_model_ids = sorted(bertopic_model_ids)

    return {"bertopic_model_ids": bertopic_model_ids}



###################### SEARCH ###########################

@esr_app.get("/esr/search_episodes_by_title/{show_key}", tags=['ES Reader'])
def search_episodes_by_title(show_key: ShowKey, title: str = None):
    '''
    Free text episode search by title
    '''
    s = esqb.search_episodes_by_title(show_key.value, title)
    es_query = s.to_dict()
    matches = esrt.return_episodes_by_title(s)
    return {"episode_count": len(matches), "episodes": matches, "es_query": es_query}


@esr_app.get("/esr/search_scenes/{show_key}", tags=['ES Reader'])
def search_scenes(show_key: ShowKey, season: str = None, episode_key: str = None, location: str = None, description: str = None):
    '''
    Facet query of nested Scene fields 
    '''
    if not (location or description):
        error = 'Unable to execute search_scenes without at least one scene property set (location or description)'
        print(error)
        return {"error": error}
    s = esqb.search_scenes(show_key.value, season=season, episode_key=episode_key, location=location, description=description)
    es_query = s.to_dict()
    matches, scene_count = esrt.return_scenes(s)
    return {"episode_count": len(matches), "scene_count": scene_count, "matches": matches, "es_query": es_query}


@esr_app.get("/esr/search_scene_events/{show_key}", tags=['ES Reader'])
def search_scene_events(show_key: ShowKey, season: str = None, episode_key: str = None, speaker: str = None, dialog: str = None, location: str = None):
    '''
    Facet query of nested Scene and SceneEvent fields 
    '''
    if not speaker and not dialog:
        return {"error": "Unable to execute search_scene_events without at least one scene_event property set: speaker or dialog"}
    s = esqb.search_scene_events(show_key.value, season=season, episode_key=episode_key, speaker=speaker, dialog=dialog)
    es_query = s.to_dict()
    matches, scene_count, scene_event_count = esrt.return_scene_events(s, location=location)
    return {"episode_count": len(matches), "scene_count": scene_count, "scene_event_count": scene_event_count, "matches": matches, "es_query": es_query}


@esr_app.get("/esr/search_scene_events_multi_speaker/{show_key}/{speakers}", tags=['ES Reader'])
def search_scene_events_multi_speaker(show_key: ShowKey, speakers: str, season: str = None, episode_key: str = None, location: str = None, intersection: bool = False):
    '''
    Facet query of Scenes comprised of SceneEvents matching 1-n speakers
    '''
    s = esqb.search_scene_events_multi_speaker(show_key.value, speakers, season=season, episode_key=episode_key)
    es_query = s.to_dict()
    matches, scene_count, scene_event_count = esrt.return_scene_events_multi_speaker(s, speakers, location=location, intersection=intersection)
    return {"episode_count": len(matches), "scene_count": scene_count, "scene_event_count": scene_event_count, "matches": matches, "es_query": es_query}


@esr_app.get("/esr/search/{show_key}", tags=['ES Reader'])
def search(show_key: ShowKey, season: str = None, episode_key: str = None, qt: str = None):
    '''
    Generic free text search of Episodes and nested Scenes and SceneEvents
    '''
    s = esqb.search_episodes(show_key.value, season=season, episode_key=episode_key, qt=qt)
    es_query = s.to_dict()
    matches, scene_count, scene_event_count = esrt.return_episodes(s)
    return {"episode_count": len(matches), "scene_count": scene_count, "scene_event_count": scene_event_count, "matches": matches, "es_query": es_query}


@esr_app.get("/esr/more_like_this/{show_key}/{episode_key}", tags=['ES Reader'])
def more_like_this(show_key: ShowKey, episode_key: str):
    s = esqb.more_like_this(show_key.value, episode_key)
    es_query = s.to_dict()
    matches = esrt.return_more_like_this(s)
    return {"similar_episode_count": len(matches), "matches": matches, "es_query": es_query}


# TODO support POST for long requests?
@esr_app.get("/esr/episode_vector_search/{show_key}", tags=['ES Reader'])
def episode_vector_search(show_key: ShowKey, qt: str, model_vendor: str = None, model_version: str = None, season: str = None):
    '''
    Generates vector embedding for qt, then determines vector cosine similarity to indexed documents using k-nearest neighbors search
    '''
    if not model_vendor:
        model_vendor = 'openai'
    if not model_version:
        model_version = 'ada002'

    if model_vendor == 'openai':
        vendor_meta = TRF_MODELS[model_vendor]
        true_model_version = vendor_meta['versions'][model_version]['true_name']
        try:
            vector_field = f'{model_vendor}_{model_version}_embeddings'
            vectorized_qt, tokens_processed_count, tokens_failed_count = ef.generate_openai_embeddings(qt, true_model_version)
            tokens_processed = []
            tokens_failed = []
        except Exception as e:
            return {"error": e}

    else:
        vendor_meta = W2V_MODELS[model_vendor]
        tag_pos = vendor_meta['pos_tag']
        try:
            # TODO normalize_and_expand_query_vocab reduced performance noticeably, disabling for now
            # qt = qp.normalize_and_expand_query_vocab(qt, show_key)
            tokenized_qt = qp.tokenize_and_remove_stopwords(qt, tag_pos=tag_pos)
            vector_field = f'{model_vendor}_{model_version}_embeddings'
            vectorized_qt, tokens_processed, tokens_failed = ef.calculate_embeddings(tokenized_qt, model_vendor, model_version)
            tokens_processed_count = len(tokens_processed)
            tokens_failed_count = len(tokens_failed)
        except Exception as e:
            return {"error": e}
        
    es_response = esqb.vector_search(show_key.value, vector_field, vectorized_qt, season=season)
    matches = esrt.return_vector_search(es_response)
    return {
        "match_count": len(matches), 
        "vector_field": vector_field, 
        "tokens_processed": tokens_processed, 
        "tokens_processed_count": tokens_processed_count, 
        "tokens_failed": tokens_failed, 
        "tokens_failed_count": tokens_failed_count, 
        "matches": matches
    }


@esr_app.get("/esr/episode_mlt_vector_search/{show_key}/{episode_key}", tags=['ES Reader'])
def episode_mlt_vector_search(show_key: ShowKey, episode_key: str, model_vendor: str = None, model_version: str = None):
    '''
    Generates vector embedding for qt, then determines vector cosine similarity to indexed documents using k-nearest neighbors search
    '''
    if not model_vendor:
        model_vendor = 'openai'
    if not model_version:
        model_version = 'ada002'

    vector_field = f'{model_vendor}_{model_version}_embeddings'
        
    s = esqb.fetch_episode_embedding(show_key.value, episode_key, vector_field)
    episode_embedding = esrt.return_embedding(s, vector_field)
        
    es_response = esqb.vector_search(show_key.value, vector_field, episode_embedding)
    matches = esrt.return_vector_search(es_response)
    matches = matches[1:] # remove episode itself from results
    return {"match_count": len(matches), "vector_field": vector_field, "matches": matches}


# def util(speaker: str, m: dict, matches_by_speaker_series_embedding: dict, all_speaker_matches: list, other_speaker_count: int, other_speaker_quota: int):
#     if other_speaker_count >= other_speaker_quota:
#         return False
#     match_speaker = m['speaker']
#     # add self-matches to list, but don't count them against other_speaker quota 
#     matches_by_speaker_series_embedding.append(m)
#     if match_speaker == speaker:
#         return True
#     other_speaker_count += 1
#     # increment match_speaker overall score  
#     if match_speaker not in all_speaker_matches:
#         all_speaker_matches[match_speaker] = 0
#     inv_rank = other_speaker_quota - other_speaker_count # eesh
#     all_speaker_matches[match_speaker] += inv_rank


@esr_app.get("/esr/speaker_mlt_vector_search/{show_key}/{speaker}", tags=['ES Reader'])
def speaker_mlt_vector_search(show_key: ShowKey, speaker: str, min_depth: bool = True, model_vendor: str = None, model_version: str = None):
    '''
    Generates vector embedding for qt, then determines vector cosine similarity to indexed documents using k-nearest neighbors search
    '''
    if not model_vendor:
        model_vendor = 'openai'
    if not model_version:
        model_version = 'ada002'

    vector_field = f'{model_vendor}_{model_version}_embeddings'
        
    # alldepth_speaker_embeddings = {}
    # if speaker_series_embeddings:
    #     alldepth_speaker_embeddings['series'] = speaker_series_embeddings
    # if season_embeddings:
    #     for season, embeddings in season_embeddings.items():
    #         alldepth_speaker_embeddings[f'S{season}'] = embeddings
    # if episode_embeddings:
    #     for episode_key, embeddings in episode_embeddings.items():
    #         alldepth_speaker_embeddings[f'E{episode_key}'] = embeddings

    '''
    - fetch all speaker1 embeddings at min_depth layer 
    - for each speaker1 embedding at min_depth, init speaker2 aggs and run vector search against speaker_embeddings_unified
        - save top X matches desc by score as matches_by_speaker_embedding[speaker_embedding_layer_key][most_similar]
        - add agg ranks and tally match_count in matches_by_speaker_embedding[speaker_embedding_layer_key][speaker_rank_aggs]
        - add agg ranks and tally match_count in agg_speaker_similarity
    - assemble two data objects:
        (1) speaker2 embeddings most similar to each speaker1 embedding across min_depth layer(s)
        (2) aggregate speaker2 similarity across all speaker1 embeddings across min_depth layer(s)
    '''
    series_embeddings, season_embeddings, episode_embeddings = esqb.fetch_speaker_embeddings(show_key.value, speaker, vector_field, min_depth=min_depth)
    
    other_speaker_quota = 30
    matches_by_speaker_series_embedding = []
    all_speaker_matches = {}
    if series_embeddings:
        other_speaker_count = 0
        matches_by_speaker_series_embedding = []
        vec_search_response = esqb.vector_search(show_key.value, vector_field, series_embeddings, index_name='speaker_embeddings_unified', min_word_count=100)
        speaker_matches = esrt.return_vector_search(vec_search_response)
        for m in speaker_matches:
            if other_speaker_count >= other_speaker_quota:
                break
            match_speaker = m['speaker']
            # add self-matches to list, but don't count them against other_speaker quota 
            matches_by_speaker_series_embedding.append(m)
            if match_speaker == speaker:
                continue
            other_speaker_count += 1
            # increment match_speaker overall score  
            if match_speaker not in all_speaker_matches:
                all_speaker_matches[match_speaker] = 0
            inv_rank = other_speaker_quota - other_speaker_count # eesh
            all_speaker_matches[match_speaker] += inv_rank

    matches_by_speaker_season_embedding = {}
    if season_embeddings:
        for season, season_embedding in season_embeddings.items():
            other_speaker_count = 0
            matches_by_speaker_season_embedding[season] = []
            vec_search_response = esqb.vector_search(show_key.value, vector_field, season_embedding, index_name='speaker_embeddings_unified', min_word_count=100)
            speaker_matches = esrt.return_vector_search(vec_search_response)
            for m in speaker_matches:
                if other_speaker_count >= other_speaker_quota:
                    break
                match_speaker = m['speaker']
                # add self-matches to list, but don't count them against other_speaker quota 
                matches_by_speaker_season_embedding[season].append(m)
                if match_speaker == speaker:
                    continue
                other_speaker_count += 1
                # increment match_speaker overall score  
                if match_speaker not in all_speaker_matches:
                    all_speaker_matches[match_speaker] = 0
                inv_rank = other_speaker_quota - other_speaker_count # eesh
                all_speaker_matches[match_speaker] += inv_rank

    matches_by_speaker_episode_embedding = {}
    if episode_embeddings:
        for episode_key, episode_embedding in episode_embeddings.items():
            other_speaker_count = 0
            matches_by_speaker_episode_embedding[episode_key] = []
            vec_search_response = esqb.vector_search(show_key.value, vector_field, episode_embedding, index_name='speaker_embeddings_unified', min_word_count=100)
            speaker_matches = esrt.return_vector_search(vec_search_response)
            for m in speaker_matches:
                if other_speaker_count >= other_speaker_quota:
                    break
                match_speaker = m['speaker']
                # add self-matches to list, but don't count them against other_speaker quota 
                matches_by_speaker_episode_embedding[episode_key].append(m)
                if match_speaker == speaker:
                    continue
                other_speaker_count += 1
                # increment match_speaker overall score  
                if match_speaker not in all_speaker_matches:
                    all_speaker_matches[match_speaker] = 0
                inv_rank = other_speaker_quota - other_speaker_count # eesh
                all_speaker_matches[match_speaker] += inv_rank
    
    '''
    matches_by_speaker_embedding = {
        <speaker_embedding_layer_key>: {
            'most_similar': [{
                <speaker_embedding_layer_key>: <score>
            }],
            'speaker_rank_aggs': {}
                <speaker>: {
                    'agg_score': <agg_score>,
                    'match_count': <match_count>
                }
            }
        }
    }

    agg_speaker_similarity = {
        <speaker>: {
            'agg_score': <agg_score>,
            'match_count': <match_count>
        }
    }
    '''

    # all_speaker_matches = {}

    # cutoff = 30
    # speaker_series_matches = {}
    # all_speaker_season_matches = {}
    # all_speaker_episode_matches = {}
    # if speaker_series_embeddings:
    #     es_speakers_response = esqb.vector_search(show_key.value, vector_field, speaker_series_embeddings, index_name='speakers')
    #     speaker_matches = esrt.return_vector_search(es_speakers_response)
    #     for m in speaker_matches[:cutoff]:
    #         match_speaker = m['speaker']
    #         if match_speaker == speaker:
    #             continue
    #         if match_speaker not in all_speaker_matches:
    #             all_speaker_matches[match_speaker] = 0
    #         rank = cutoff - m['rank']
    #         all_speaker_matches[match_speaker] += rank

    # cutoff = 10
    # # all_speaker_matches = {}
    # all_speaker_season_matches = {}
    # all_speaker_episode_matches = {}
    # for key, speaker_embeddings in alldepth_speaker_embeddings.items():
    #     # compare to series-level embeddings
    #     es_speakers_response = esqb.vector_search(show_key.value, vector_field, speaker_embeddings, index_name='speakers')
    #     speaker_matches = esrt.return_vector_search(es_speakers_response)
    #     for m in speaker_matches[:cutoff]:
    #         match_speaker = m['speaker']
    #         if match_speaker == speaker:
    #             continue
    #         if match_speaker not in all_speaker_matches:
    #             all_speaker_matches[match_speaker] = 0
    #         rank = cutoff - m['rank']
    #         all_speaker_matches[match_speaker] += rank

    #     # compare to season-level embeddings
    #     es_speaker_seasons_response = esqb.vector_search(show_key.value, vector_field, speaker_embeddings, index_name='speaker_seasons')
    #     speaker_season_matches = esrt.return_vector_search(es_speaker_seasons_response)
    #     for m in speaker_season_matches[:cutoff]:
    #         match_speaker = m['speaker']
    #         if match_speaker == speaker:
    #             continue
    #         if match_speaker not in all_speaker_season_matches:
    #             all_speaker_season_matches[match_speaker] = dict(agg_score=0, sources=[])
    #         rank = cutoff - m['rank']
    #         all_speaker_season_matches[match_speaker]['agg_score'] += rank
    #         all_speaker_season_matches[match_speaker]['sources'].append(f"{key}:{m['season']}")

    #     # compare to season-level embeddings
    #     es_speaker_episodes_response = esqb.vector_search(show_key.value, vector_field, speaker_embeddings, index_name='speaker_episodes')
    #     speaker_episode_matches = esrt.return_vector_search(es_speaker_episodes_response)
    #     for m in speaker_episode_matches[:cutoff]:
    #         match_speaker = m['speaker']
    #         if match_speaker == speaker:
    #             continue
    #         if match_speaker not in all_speaker_episode_matches:
    #             all_speaker_episode_matches[match_speaker] = dict(agg_score=0, sources=[])
    #         rank = cutoff - m['rank']
    #         all_speaker_episode_matches[match_speaker]['agg_score'] += rank
    #         all_speaker_episode_matches[match_speaker]['sources'].append(f"{key}:{m['episode_key']}")
    
    all_speaker_matches = sorted(all_speaker_matches.items(), key=lambda kv: kv[1], reverse=True)
    all_speaker_matches = {asm[0]:asm[1] for asm in all_speaker_matches}

    return {
        "all_speaker_matches": all_speaker_matches, 
        "matches_by_speaker_series_embedding": matches_by_speaker_series_embedding,
        "matches_by_speaker_season_embedding": matches_by_speaker_season_embedding, 
        "matches_by_speaker_episode_embedding": matches_by_speaker_episode_embedding
    }


@esr_app.get("/esr/episode_topic_vector_search/{show_key}/{episode_key}/{topic_grouping}", tags=['ES Reader'])
def episode_topic_vector_search(show_key: ShowKey, episode_key: str, topic_grouping: str, model_vendor: str = None, model_version: str = None):
    '''
    Fetches vector embedding for episode, then determines vector cosine similarity to indexed topics using k-nearest neighbors search
    '''
    if not model_vendor:
        model_vendor = 'openai'
    if not model_version:
        model_version = 'ada002'

    vector_field = f'{model_vendor}_{model_version}_embeddings'
        
    s = esqb.fetch_episode_embedding(show_key.value, episode_key, vector_field)
    episode_embedding = esrt.return_embedding(s, vector_field)
        
    es_response = esqb.topic_vector_search(topic_grouping, vector_field, episode_embedding)
    topics = esrt.return_vector_search(es_response)
    return {"topic_count": len(topics), "vector_field": vector_field, "topics": topics}


@esr_app.get("/esr/topic_episode_vector_search/{topic_grouping}/{topic_key}/{show_key}", tags=['ES Reader'])
def topic_episode_vector_search(topic_grouping: str, topic_key: str, show_key: ShowKey, model_vendor: str = None, model_version: str = None):
    '''
    Fetches vector embedding for topic, then determines vector cosine similarity to indexed episodes using k-nearest neighbors search
    '''
    print(f'begin topic_episode_vector_search for topic_grouping={topic_grouping} topic_key={topic_key} show_key={show_key}')
    if not model_vendor:
        model_vendor = 'openai'
    if not model_version:
        model_version = 'ada002'

    vector_field = f'{model_vendor}_{model_version}_embeddings'
        
    s = esqb.fetch_topic_embedding(topic_grouping, topic_key, vector_field)
    topic_embedding = esrt.return_embedding(s, vector_field)
    if not topic_embedding:
        return {"error": f"Unable to run `topic_episode_vector_search`: No embeddings for topic_grouping={topic_grouping} topic_key={topic_key} vector_field={vector_field}"}
        
    es_response = esqb.vector_search(show_key, vector_field, topic_embedding)
    episodes = esrt.return_vector_search(es_response)
    return {"episodes_count": len(episodes), "vector_field": vector_field, "episodes": episodes}


@esr_app.get("/esr/speaker_topic_vector_search/{show_key}/{speaker}/{topic_grouping}", tags=['ES Reader'])
def speaker_topic_vector_search(show_key: ShowKey, speaker: str, topic_grouping: str, model_vendor: str = None, model_version: str = None,
                                seasons: str = None, episode_keys: str = None, min_depth: bool = False):
    '''
    Fetches (does not generate) vector embedding for speaker, then determines vector cosine similarity to indexed topics using k-nearest neighbors search
    '''
    if not model_vendor:
        model_vendor = 'openai'
    if not model_version:
        model_version = 'ada002'

    # this got hella ugly
    if seasons:
        seasons = seasons.split(',')
    if episode_keys:
        episode_keys = episode_keys.split(',')
    seasons_only = False
    episodes_only = False
    if seasons and not episode_keys:
        seasons_only = True
    elif episode_keys and not seasons:
        episodes_only = True

    vector_field = f'{model_vendor}_{model_version}_embeddings'

    # series_embeddings, season_embeddings, episode_embeddings = esqb.fetch_speaker_embeddings(
    #     show_key.value, speaker, vector_field, seasons=seasons, episode_keys=episode_keys, min_depth=min_depth)
    
    series_topics = []
    season_topics = {}
    episode_topics = {}
    
    es_speaker_response = fetch_speaker(show_key, speaker, include_seasons=True, include_episodes=True)
    if 'speaker' not in es_speaker_response:
        return {"error": f"Failed to speaker_topic_vector_search for show_key={show_key.value} speaker={speaker}: speaker lookup failed"}
    es_speaker = es_speaker_response['speaker']

    # speaker_series_embeddings = getattr(es_speaker, vector_field)
    # NOTE 9/26/2024 I'm grossed out by how complicated this endpoint is. And I just made it more complicated by forcing it to do the simple
    # thing I need it to do right now, which is run a topic vector search for a single speaker_episode against a topic. I don't think there's
    # any other endpoint that's close to having that capability.
    if vector_field in es_speaker and not (episode_keys or seasons):
        s = esqb.topic_vector_search(topic_grouping, vector_field, es_speaker[vector_field])
        series_topics = esrt.return_vector_search(s)

    if 'episodes' in es_speaker and not seasons_only:
        for es_speaker_episode in es_speaker['episodes']:
            if episode_keys and es_speaker_episode['episode_key'] not in episode_keys:
                continue
            if vector_field in es_speaker_episode:
                s = esqb.topic_vector_search(topic_grouping, vector_field, es_speaker_episode[vector_field])
                topics = esrt.return_vector_search(s)
                if topics:
                    episode_topics[es_speaker_episode['episode_key']] = topics

    if 'seasons' in es_speaker and not episodes_only:
        for es_speaker_season in es_speaker['seasons']:
            if seasons and es_speaker_season['season'] not in seasons:
                continue
            if vector_field in es_speaker_season:
                s = esqb.topic_vector_search(topic_grouping, vector_field, es_speaker_season[vector_field])
                topics = esrt.return_vector_search(s)
                if topics:
                    season_topics[es_speaker_season['season']] = topics

    return {"series_topics": series_topics, "season_topics": season_topics, "episode_topics": episode_topics}


@DeprecationWarning
# TODO should this be deprecated, or just discouraged? Used as a fallback when `topic_speaker_search` fails?
@esr_app.get("/esr/topic_speaker_vector_search/{topic_grouping}/{topic_key}/{show_key}", tags=['ES Reader'])
def topic_speaker_vector_search(topic_grouping: str, topic_key: str, show_key: ShowKey, model_vendor: str = None, model_version: str = None):
    '''
    Fetches vector embedding for topic, then determines vector cosine similarity to indexed speakers using k-nearest neighbors search
    '''
    if not model_vendor:
        model_vendor = 'openai'
    if not model_version:
        model_version = 'ada002'

    vector_field = f'{model_vendor}_{model_version}_embeddings'
        
    s = esqb.fetch_topic_embedding(topic_grouping, topic_key, vector_field)
    topic_embedding = esrt.return_embedding(s, vector_field)
    if not topic_embedding:
        return {"error": f"Unable to run `topic_speaker_vector_search`: No embeddings for topic_grouping={topic_grouping} topic_key={topic_key} vector_field={vector_field}"}
        
    # TODO this only searches speakers who have series-level embeddings, needs work
    es_response = esqb.vector_search(show_key, vector_field, topic_embedding, index_name='speakers')
    speakers = esrt.return_vector_search(es_response)
    return {"speakers_count": len(speakers), "vector_field": vector_field, "speakers": speakers}


@DeprecationWarning
@esr_app.get("/esr/topic_speaker_search/{topic_grouping}/{topic_key}", tags=['ES Reader'])
def topic_speaker_search(topic_grouping: str, topic_key: str, show_key: ShowKey = None, min_word_count: int = None):
    '''
    Search speakers by topic mapping
    '''                
    if show_key:
        show_key = show_key.value

    topic_response = fetch_topic(topic_grouping, topic_key)
    if 'topic' not in topic_response:
        return {"error": f"Unable to run `topic_speaker_search`: No topic found for topic_grouping={topic_grouping} topic_key={topic_key}"}
    topic = topic_response['topic']

    if 'parent_key' not in topic or topic['parent_key'] == '':
        is_parent = True
    else:
        is_parent = False

    s = esqb.search_speakers_by_topic(topic_grouping, topic_key, is_parent=is_parent, show_key=show_key, min_word_count=min_word_count)
    es_query = s.to_dict()
    speakers = esrt.return_speakers(s)
    return {"speakers_count": len(speakers), "is_parent_topic": is_parent, "speakers": speakers, "es_query": es_query}


# @esr_app.get("/esr/test_vector_search/{show_key}", tags=['ES Reader'])
# def test_vector_search(show_key: ShowKey, qt: str, model_vendor: str = None, model_version: str = None, normalize_and_expand: bool = False):
#     '''
#     Experimental endpoint for troubleshooting ontology overrides and other qt alterations preceding vectorization
#     '''
#     if not model_vendor:
#         model_vendor = 'webvectors'
#     if not model_version:
#         model_version = '223'

#     # NOTE currently only set up for word2vec, not for openai embeddings

#     vendor_meta = W2V_MODELS[model_vendor]
#     tag_pos = vendor_meta['pos_tag']

#     try:
#         if normalize_and_expand:
#             qt = qp.normalize_and_expand_query_vocab(qt, show_key)
#         tokenized_qt = qp.tokenize_and_remove_stopwords(qt, tag_pos=tag_pos)
#     except Exception as e:
#         return {"error": e}
#     return {"normd_expanded_qt": qt, "tokenized_qt": tokenized_qt}


@esr_app.get("/esr/search_speakers/{qt}/", tags=['ES Reader'])
def search_speakers(qt: str, show_key: ShowKey = None, extra_fields: str = None):
    '''
    Search for a speaker by query term
    '''                
    if show_key:
        show_key = show_key.value
    return_fields = ['speaker', 'alt_names', 'actor_names', 'season_count', 'episode_count', 'scene_count', 'line_count', 'word_count', 'openai_ada002_word_count']
    if extra_fields:
        extra_fields = extra_fields.split(',')
        return_fields.extend(extra_fields)
    s = esqb.search_speakers(qt, show_key=show_key, return_fields=return_fields)
    es_query = s.to_dict()
    speaker_matches = esrt.return_speakers(s)
    return {"speaker_matches": speaker_matches, "es_query": es_query}


@esr_app.get("/esr/find_episodes_by_topic/{show_key}/{topic_grouping}/{topic_key}", tags=['ES Reader'])
def find_episodes_by_topic(show_key: ShowKey, topic_grouping: str, topic_key: str, season: int = None, sort_by: str = None):
    '''
    Search episodes by topic
    '''                
    if not sort_by:
        sort_by = 'score'
    s = esqb.search_episode_topics(show_key, topic_grouping, topic_key, season=season, sort_by=sort_by)
    es_query = s.to_dict()
    episode_topics = esrt.return_topics(s)
    return {"episode_topics": episode_topics, "es_query": es_query}


@esr_app.get("/esr/find_speakers_by_topic/{topic_grouping}/{topic_key}", tags=['ES Reader'])
def find_speakers_by_topic(topic_grouping: str, topic_key: str, show_key: ShowKey = None, min_word_count: int = None):
    '''
    Search speakers by topic. Not restricted to a given show. 
    '''    
    if show_key:
        show_key = show_key.value
    s = esqb.search_speaker_topics(topic_grouping, topic_key, show_key=show_key, min_word_count=min_word_count)
    es_query = s.to_dict()
    speaker_topics = esrt.return_topics(s)
    return {"speaker_topics": speaker_topics, "es_query": es_query}


@esr_app.get("/esr/find_speaker_seasons_by_topic/{topic_grouping}/{topic_key}/{show_key}", tags=['ES Reader'])
def find_speaker_seasons_by_topic(topic_grouping: str, topic_key: str, show_key: ShowKey, season: int = None, min_word_count: int = None):
    '''
    Search speaker_seasons by topic 
    '''    
    s = esqb.search_speaker_season_topics(topic_grouping, topic_key, show_key.value, season=season, min_word_count=min_word_count)
    es_query = s.to_dict()
    speaker_season_topics = esrt.return_topics(s)
    return {"speaker_season_topics": speaker_season_topics, "es_query": es_query}


@esr_app.get("/esr/find_speaker_episodes_by_topic/{topic_grouping}/{topic_key}/{show_key}", tags=['ES Reader'])
def find_speaker_episodes_by_topic(topic_grouping: str, topic_key: str, show_key: ShowKey, season: int = None, episode_key: str = None, min_word_count: int = None):
    '''
    Search speaker_episodes by topic 
    '''    
    s = esqb.search_speaker_episode_topics(topic_grouping, topic_key, show_key.value, season=season, episode_key=episode_key, min_word_count=min_word_count)
    es_query = s.to_dict()
    speaker_episode_topics = esrt.return_topics(s)
    return {"speaker_episode_topics": speaker_episode_topics, "es_query": es_query}



###################### AGGREGATIONS ###########################

@esr_app.get("/esr/agg_seasons/{show_key}", tags=['ES Reader'])
def agg_seasons(show_key: ShowKey, location: str = None):
    s = esqb.agg_seasons(show_key.value, location=location)
    es_query = s.to_dict()
    season_count = esrt.return_season_count(s)
    return {"season_count": season_count, "es_query": es_query}


@esr_app.get("/esr/agg_seasons_by_speaker/{show_key}", tags=['ES Reader'])
def agg_seasons_by_speaker(show_key: ShowKey, location: str = None):
    s = esqb.agg_seasons_by_speaker(show_key.value, location=location)
    es_query = s.to_dict()
    season_count = agg_seasons(show_key, location=location)
    matches = esrt.return_seasons_by_speaker(s, season_count['season_count'], location=location)
    return {"speaker_count": len(matches), "seasons_by_speaker": matches, "es_query": es_query}


@esr_app.get("/esr/agg_seasons_by_location/{show_key}", tags=['ES Reader'])
def agg_seasons_by_location(show_key: ShowKey):
    s = esqb.agg_seasons_by_location(show_key.value)
    es_query = s.to_dict()
    season_count = agg_seasons(show_key)
    matches = esrt.return_seasons_by_location(s, season_count['season_count'])
    return {"location_count": len(matches), "seasons_by_location": matches, "es_query": es_query}


@esr_app.get("/esr/agg_episodes/{show_key}", tags=['ES Reader'])
def agg_episodes(show_key: ShowKey, season: str = None, location: str = None):
    s = esqb.agg_episodes(show_key.value, season=season, location=location)
    es_query = s.to_dict()
    episode_count = esrt.return_episode_count(s)
    return {"episode_count": episode_count, "es_query": es_query}


@esr_app.get("/esr/agg_episodes_by_speaker/{show_key}", tags=['ES Reader'])
def agg_episodes_by_speaker(show_key: ShowKey, season: str = None, location: str = None, other_speaker: str = None):
    s = esqb.agg_episodes_by_speaker(show_key.value, season=season, location=location, other_speaker=other_speaker)
    es_query = s.to_dict()
    # separate call to get episode_count without double-counting per speaker
    episode_count = agg_episodes(show_key, season=season, location=location)
    matches = esrt.return_episodes_by_speaker(s, episode_count['episode_count'], location=location, other_speaker=other_speaker)
    return {"speaker_count": len(matches), "episodes_by_speaker": matches, "es_query": es_query}


@esr_app.get("/esr/agg_episodes_by_location/{show_key}", tags=['ES Reader'])
def agg_episodes_by_location(show_key: ShowKey, season: str = None):
    s = esqb.agg_episodes_by_location(show_key.value, season=season)
    es_query = s.to_dict()
    # separate call to get episode_count without double-counting per speaker
    episode_count = agg_episodes(show_key, season=season)
    matches = esrt.return_episodes_by_location(s, episode_count['episode_count'])
    return {"location_count": len(matches), "episodes_by_location": matches, "es_query": es_query}


@esr_app.get("/esr/agg_scenes/{show_key}", tags=['ES Reader'])
def agg_scenes(show_key: ShowKey, season: str = None, episode_key: str = None, location: str = None):
    s = esqb.agg_scenes(show_key.value, season=season, episode_key=episode_key, location=location)
    es_query = s.to_dict()
    scene_count = esrt.return_scene_count(s)
    return {"scene_count": scene_count, "es_query": es_query}


@esr_app.get("/esr/agg_scenes_by_speaker/{show_key}", tags=['ES Reader'])
def agg_scenes_by_speaker(show_key: ShowKey, season: str = None, episode_key: str = None, location: str = None, other_speaker: str = None):
    s = esqb.agg_scenes_by_speaker(show_key.value, season=season, episode_key=episode_key, location=location, other_speaker=other_speaker)
    es_query = s.to_dict()
    # separate call to get scene_count without double-counting per speaker
    scene_count = agg_scenes(show_key, season=season, episode_key=episode_key, location=location)
    matches = esrt.return_scenes_by_speaker(s, scene_count['scene_count'], location=location, other_speaker=other_speaker)
    return {"speaker_count": len(matches), "scenes_by_speaker": matches, "es_query": es_query}


@esr_app.get("/esr/agg_scenes_by_location/{show_key}", tags=['ES Reader'])
def agg_scenes_by_location(show_key: ShowKey, season: str = None, episode_key: str = None, speaker: str = None):
    s = esqb.agg_scenes_by_location(show_key.value, season=season, episode_key=episode_key, speaker=speaker)
    es_query = s.to_dict()
    matches = esrt.return_scenes_by_location(s, speaker=speaker)
    return {"location_count": len(matches), "scenes_by_location": matches, "es_query": es_query}


@esr_app.get("/esr/agg_scene_events_by_speaker/{show_key}", tags=['ES Reader'])
def agg_scene_events_by_speaker(show_key: ShowKey, season: str = None, episode_key: str = None, dialog: str = None):
    s = esqb.agg_scene_events_by_speaker(show_key.value, season=season, episode_key=episode_key, dialog=dialog)
    es_query = s.to_dict()
    matches = esrt.return_scene_events_by_speaker(s, dialog=dialog)
    return {"speaker_count": len(matches), "scene_events_by_speaker": matches, "es_query": es_query}


@esr_app.get("/esr/agg_dialog_word_counts/{show_key}", tags=['ES Reader'])
def agg_dialog_word_counts(show_key: ShowKey, season: str = None, episode_key: str = None, speaker: str = None):
    s = esqb.agg_dialog_word_counts(show_key.value, season=season, episode_key=episode_key, speaker=speaker)
    es_query = s.to_dict()
    matches = esrt.return_dialog_word_counts(s, speaker=speaker)
    return {"dialog_word_counts": matches, "es_query": es_query}


# TODO this might be going away as it's largely replaced by speaker index
@esr_app.get("/esr/composite_speaker_aggs/{show_key}", tags=['ES Reader'])
def composite_speaker_aggs(show_key: ShowKey, season: str = None, episode_key: str = None):
    if not season and not episode_key:
        speaker_season_counts = agg_seasons_by_speaker(show_key)
    if not episode_key:
        speaker_episode_counts = agg_episodes_by_speaker(show_key, season=season)
    speaker_scene_counts = agg_scenes_by_speaker(show_key, season=season, episode_key=episode_key)
    speaker_line_counts = agg_scene_events_by_speaker(show_key, season=season, episode_key=episode_key)
    speaker_word_counts = agg_dialog_word_counts(show_key, season=season, episode_key=episode_key)

    # TODO refactor this to generically handle dicts threading together
    speakers = {}
    if not season and not episode_key:
        for speaker, season_count in speaker_season_counts['seasons_by_speaker'].items():
            if speaker not in speakers:
                speakers[speaker] = {}
                speakers[speaker]['speaker'] = speaker
            speakers[speaker]['season_count'] = season_count
    if not episode_key:
        for speaker, episode_count in speaker_episode_counts['episodes_by_speaker'].items():
            if speaker not in speakers:
                speakers[speaker] = {}
                speakers[speaker]['speaker'] = speaker
            speakers[speaker]['episode_count'] = episode_count
    for speaker, scene_count in speaker_scene_counts['scenes_by_speaker'].items():
        if speaker not in speakers:
            speakers[speaker] = {}
            speakers[speaker]['speaker'] = speaker
        speakers[speaker]['scene_count'] = scene_count
    for speaker, line_count in speaker_line_counts['scene_events_by_speaker'].items():
        if speaker not in speakers:
            speakers[speaker] = {}
            speakers[speaker]['speaker'] = speaker
        speakers[speaker]['line_count'] = line_count
    for speaker, word_count in speaker_word_counts['dialog_word_counts'].items():
        if speaker not in speakers:
            speakers[speaker] = {}
            speakers[speaker]['speaker'] = speaker
        speakers[speaker]['word_count'] = int(word_count)  # NOTE not sure why casting is needed here

    # TODO shouldn't I be able to sort on a key for a dict within a dict
    speaker_dicts = speakers.values()
    sort_field = 'scene_count'
    if not episode_key:
        sort_field = 'episode_count'
    speaker_agg_composite = sorted(speaker_dicts, key=itemgetter(sort_field), reverse=True)

    return {"speaker_count": len(speaker_agg_composite), "speaker_agg_composite": speaker_agg_composite} 


@esr_app.get("/esr/composite_location_aggs/{show_key}", tags=['ES Reader'])
def composite_location_aggs(show_key: ShowKey, season: str = None, episode_key: str = None):
    if not season and not episode_key:
        location_season_counts = agg_seasons_by_location(show_key)
    if not episode_key:
        location_episode_counts = agg_episodes_by_location(show_key, season=season)
    location_scene_counts = agg_scenes_by_location(show_key, season=season, episode_key=episode_key)

    # TODO refactor this to generically handle dicts threading together
    locations = {}
    if not season and not episode_key:
        for location, season_count in location_season_counts['seasons_by_location'].items():
            if location not in locations:
                locations[location] = {}
                locations[location]['location'] = location
            locations[location]['season_count'] = season_count
    if not episode_key:
        for location, episode_count in location_episode_counts['episodes_by_location'].items():
            if location not in locations:
                locations[location] = {}
                locations[location]['location'] = location
            locations[location]['episode_count'] = episode_count
    for location, scene_count in location_scene_counts['scenes_by_location'].items():
        if location not in locations:
            locations[location] = {}
            locations[location]['location'] = location
        locations[location]['scene_count'] = scene_count

    # TODO shouldn't I be able to sort on a key for a dict within a dict
    location_dicts = locations.values()
    sort_field = 'scene_count'
    if not episode_key:
        sort_field = 'episode_count'
    location_agg_composite = sorted(location_dicts, key=itemgetter(sort_field), reverse=True)

    return {"location_count": len(location_agg_composite), "location_agg_composite": location_agg_composite} 


@esr_app.get("/esr/agg_numeric_distrib_into_percentiles/{show_key}/{index}/{numeric_field}", tags=['ES Reader'])
def agg_numeric_distrib_into_percentiles(show_key: ShowKey, index: str, numeric_field: str, constraints: str = None):
    '''
    Slice up the distribution of values for a numeric field into 100 percentile slots, so it's easier to compare those values
    Motivation: KNN similarity of episode topics, ranging from 81-91, with most results between 84-86, being difficult to usefully compare in a pie chart
    TODO this invalidates the current `speaker_episode_topics.score` field and probably similar fields elsewhere, should it be:
        (a) incorporated as part of the existing bulk speaker_episode_topics index populator function
        (b) added as a new downstream dependency in the broader speaker_episode_topics index populating pipeline
        (c) invoked in real time rather than baked into an index
    '''

    constraints_dict = {}
    if constraints:
        constraint_bits = constraints.split(',')
        for c_bit in constraint_bits:
            c_kv = c_bit.split(':')
            if len(c_kv) != 2:
                print(f'Malformed constraint={c_kv}, must have `key`:`value` format. Ignoring.')
                continue
            constraints_dict[c_kv[0]] = c_kv[1]

    s = esqb.agg_numeric_distrib_into_percentiles(show_key.value, index, numeric_field, constraints=constraints_dict)
    es_query = s.to_dict()
    percentile_distribution = esrt.return_numeric_distrib_into_percentiles(s, numeric_field)

    return {"percentile_distribution": percentile_distribution, "es_query": es_query}
        


###################### RELATIONS ###########################

@esr_app.get("/esr/episode_relations_graph/{show_key}/{model_vendor}/{model_version}", tags=['ES Reader'])
def episode_relations_graph(show_key: ShowKey, model_vendor: str, model_version: str, max_edges: int = None, season: str = None):
    if not max_edges:
        max_edges = 5
    # nodes 
    episodes_and_relations = fetch_all_episode_relations(show_key, model_vendor, model_version)
    nodes = []
    links = []
    episode_keys_to_indexes = {}
    relations_field = f'{model_vendor}_{model_version}_relations_text'
    i = 0
    for episode in episodes_and_relations['episode_relations']:
        node = {}
        node['name'] = episode['title']
        node['episode_key'] = episode['episode_key']
        node['group'] = episode['season']
        nodes.append(node)
        episode_keys_to_indexes[episode['episode_key']] = i
        i += 1
    
    # second loop is needed since we don't know the index positions of every episode during the first loop
    for episode in episodes_and_relations['episode_relations']:
        if relations_field not in episode:
            continue
        relations_count = min(len(episode[relations_field]), max_edges)
        for i in range(relations_count):
            link = {}
            target_episode_key =  episode[relations_field][i].split('|')[0]
            link['source'] = episode_keys_to_indexes[episode['episode_key']]
            link['target'] = episode_keys_to_indexes[target_episode_key]
            link['value'] = max_edges - i
            links.append(link)
    
    return {"nodes": nodes, "links": links}


@esr_app.get("/esr/speaker_relations_graph/{show_key}/{episode_key}", tags=['ES Reader'])
def speaker_relations_graph(show_key: ShowKey, episode_key: str):
    
    episode_response = fetch_episode(show_key, episode_key)
    episode = episode_response['es_episode']

    episode_speakers_response = fetch_speakers_for_episode(show_key, episode_key)
    speakers = episode_speakers_response['speaker_episodes']
    speakers_to_node_i = {s['speaker']:i for i, s in enumerate(speakers)}
    speaker_associations = {s['speaker']:set() for i, s in enumerate(speakers)}

    edges = []
    source_targets_to_edge_i = {}
    
    for i, scene in enumerate(episode['scenes']):
        if 'scene_events' not in scene:
            continue
        speakers_in_scene = set()
        speaker_node_ids_in_scene = set()
        for scene_event in scene['scene_events']:
            if 'spoken_by' in scene_event:
                speakers_in_scene.add(scene_event['spoken_by'])
        for spkr in speakers_in_scene:
            if spkr not in speakers_to_node_i:
                continue
            speaker_node_ids_in_scene.add(speakers_to_node_i[spkr])
            other_speakers = set(speakers_in_scene)
            other_speakers.remove(spkr)
            speaker_associations[spkr].update(other_speakers)
        for source_speaker_node_i in speaker_node_ids_in_scene:
            for target_speaker_node_i in speaker_node_ids_in_scene:
                if source_speaker_node_i == target_speaker_node_i:
                    continue
                source_target = f'{source_speaker_node_i}_{target_speaker_node_i}'
                if source_target not in source_targets_to_edge_i:
                    edges.append({'source': source_speaker_node_i, 'target': target_speaker_node_i, 'value': 1})
                    source_targets_to_edge_i[source_target] = len(edges)-1
                else:
                    edge_i = source_targets_to_edge_i[source_target]
                    edges[edge_i]['value'] += 1

    for s in speakers:
        s['associations'] = list(speaker_associations[s['speaker']])

    return {"nodes": speakers, "edges": edges}



###################### OTHER ###########################

@esr_app.get("/esr/keywords_by_episode/{show_key}/{episode_key}", tags=['ES Reader'])
def keywords_by_episode(show_key: ShowKey, episode_key: str, exclude_speakers: bool = False):
    response = esqb.keywords_by_episode(show_key.value, episode_key)
    all_speakers = []
    if exclude_speakers:
        res = agg_scenes_by_speaker(show_key, episode_key=episode_key) # TODO should this use agg_episodes_by_speaker now?
        all_speakers = res['scenes_by_speaker'].keys()
    matches = esrt.return_keywords_by_episode(response, exclude_terms=all_speakers)
    return {"keyword_count": len(matches), "keywords": matches}


@esr_app.get("/esr/keywords_by_corpus/{show_key}", tags=['ES Reader'])
def keywords_by_corpus(show_key: ShowKey, season: str = None, exclude_speakers: bool = False):
    response = esqb.keywords_by_corpus(show_key.value, season=season)
    all_speakers = []
    if exclude_speakers:
        res = agg_episodes_by_speaker(show_key, season=season)
        all_speakers = res['episodes_by_speaker'].keys()
    matches = esrt.return_keywords_by_corpus(response, exclude_terms=all_speakers)
    return {"keyword_count": len(matches), "keywords": matches}


@esr_app.get("/esr/generate_episode_gantt_sequence/{show_key}/{episode_key}", tags=['ES Reader'])
def generate_episode_gantt_sequence(show_key: ShowKey, episode_key: str):
    '''
    TODO
    '''
    max_line_chars = 280
    
    dialog_timeline = []
    location_timeline = []
    word_i = 0
    scene_start_i = 0
    # fetch episode data
    episode = fetch_episode(show_key, episode_key)
    es_episode = episode['es_episode']
    if 'scenes' not in es_episode:
        return {"dialog_timeline": [], "location_timeline": []}
    # for each scene containing dialog:
    #   - for each dialog scene_event, add a dialog_span specifying speaker and start/end word index of dialog
    #   - add a location_span specifying location and start/end word index of scene
    for i, s in enumerate(es_episode['scenes']):
        if 'scene_events' not in s:
            continue
        scene_lines = []
        for j, se in enumerate(s['scene_events']):
            if 'spoken_by' and 'dialog' in se:
                line_dialog = se['dialog']
                line_wc = len(line_dialog.split())
                if len(line_dialog) > max_line_chars:
                    line_dialog = f'{line_dialog[:max_line_chars]}...'
                dialog_span = dict(Task=se['spoken_by'], Start=word_i, Finish=(word_i+line_wc-1), Line=line_dialog, scene=i, scene_event=j)
                dialog_timeline.append(dialog_span)
                word_i += line_wc
                scene_lines.append(f"{se['spoken_by']}: {line_dialog}")
        location_span = dict(Task=s['location'], Start=scene_start_i, Finish=(word_i-1), Line='<br>'.join(scene_lines), scene=i)
        location_timeline.append(location_span)
        scene_start_i = word_i

    return {"dialog_timeline": dialog_timeline, "location_timeline": location_timeline}


@esr_app.get("/esr/generate_series_speaker_gantt_sequence/{show_key}", tags=['ES Reader'])
def generate_series_speaker_gantt_sequence(show_key: ShowKey, limit_cast: bool = False, overwrite_file: bool = False):
    '''
    TODO 
    '''
    episodes_to_speaker_line_counts = {}
    episode_speakers_sequence = []
    
    # get ordered list of all episodes
    response = fetch_simple_episodes(show_key)
    episodes = response['episodes']

    # for each episode:
    # - fetch all speakers ordered by scene_event count (how many lines they have)
    # - transform results into lists of span dicts for creating plotly gantt charts
    episode_i = 0
    for episode in episodes:
        episode_key = episode['episode_key']
        episode_title = episode['title']
        episode_season = episode['season']
        sequence_in_season = episode['sequence_in_season']

        # fetch speakers and line counts
        response = agg_scene_events_by_speaker(show_key, episode_key=episode_key)
        speaker_line_counts = response['scene_events_by_speaker']
        del speaker_line_counts['_ALL_']
        episodes_to_speaker_line_counts[episode_key] = speaker_line_counts
        # transform speakers/line counts to plotly-gantt-friendly span dicts
        for speaker, line_count in speaker_line_counts.items():
            speaker_span = dict(Task=speaker, Start=episode_i, Finish=(episode_i+1), episode_key=episode_key, episode_title=episode_title, 
                                count=line_count, season=episode_season, sequence_in_season=sequence_in_season,
                                info=f'{episode_title} ({line_count} lines)')
            episode_speakers_sequence.append(speaker_span)

        episode_i += 1

    # TODO move this to fig_builder? (where it has to filter rows from the df)
    if limit_cast:
        trimmed_episode_speakers_sequence = []
        for d in episode_speakers_sequence:
            if d['Task'] in show_metadata[show_key]['regular_cast'].keys() or d['Task'] in show_metadata[show_key]['recurring_cast'].keys():
                trimmed_episode_speakers_sequence.append(d)
        episode_speakers_sequence = trimmed_episode_speakers_sequence

    if overwrite_file:
        file_path = f'./app/data/speaker_gantt_sequence_{show_key}.csv'
        print(f'writing speaker gantt sequence dataframe to file_path={file_path}')
        df = pd.DataFrame(episode_speakers_sequence)
        df.to_csv(file_path)

    return {"episodes_to_speaker_line_counts": episodes_to_speaker_line_counts, 
            "episode_speakers_sequence": episode_speakers_sequence}


@esr_app.get("/esr/generate_series_location_gantt_sequence/{show_key}", tags=['ES Reader'])
def generate_series_location_gantt_sequence(show_key: ShowKey, overwrite_file: bool = False):
    '''
    TODO 
    '''
    episodes_to_location_counts = {}
    episode_locations_sequence = []

    # limit the superset of locations to those occurring in at least 3 episodes
    response = agg_episodes_by_location(show_key)
    location_episode_counts = response['episodes_by_location']
    del location_episode_counts['_ALL_']
    recurring_locations = [location for location, episode_count in location_episode_counts.items() if episode_count > 2]
    
    # get ordered list of all episodes
    response = fetch_simple_episodes(show_key)
    episodes = response['episodes']

    # for each episode:
    # - fetch all speakers ordered by scene_event count (how many lines they have)
    # - fetch all locations ordered by scene count
    # - transform results of both into lists of span dicts for creating plotly gantt charts
    episode_i = 0
    for episode in episodes:
        episode_key = episode['episode_key']
        episode_title = episode['title']
        episode_season = episode['season']
        sequence_in_season = episode['sequence_in_season']

        # fetch locations and scene counts
        response = agg_scenes_by_location(show_key, episode_key=episode_key)
        location_counts = response['scenes_by_location']
        del location_counts['_ALL_']
        episodes_to_location_counts[episode_key] = location_counts
        # transform locations/counts to plotly-gantt-friendly span dicts
        for location, scene_count in location_counts.items():
            if location in recurring_locations:
                location_span = dict(Task=location, Start=episode_i, Finish=(episode_i+1), episode_key=episode_key, episode_title=episode_title, 
                                     count=scene_count, season=episode_season, sequence_in_season=sequence_in_season,
                                     info=f'{episode_title} ({scene_count} scenes)')
                episode_locations_sequence.append(location_span)

        episode_i += 1

    if overwrite_file:
        file_path = f'./app/data/location_gantt_sequence_{show_key}.csv'
        print(f'writing location gantt sequence dataframe to file_path={file_path}')
        df = pd.DataFrame(episode_locations_sequence)
        df.to_csv(file_path)

    return {"episodes_to_location_counts": episodes_to_location_counts,
            "episode_locations_sequence": episode_locations_sequence}


@esr_app.get("/esr/generate_series_topic_gantt_sequence/{show_key}", tags=['ES Reader'])
def generate_series_topic_gantt_sequence(show_key: ShowKey, topic_grouping: str = None, topic_threshold: int = None, model_vendor: str = None, model_version: str = None,
                                         overwrite_file: bool = False):
    '''
    TODO Generate composite of all scene_event aggs per speaker for each individual episode
    '''
    if not topic_grouping:
        topic_grouping = EPISODE_TOPIC_GROUPINGS[0]
    if not topic_threshold:
        topic_threshold = 20
    if not model_vendor:
        model_vendor = 'openai'
    if not model_version:
        model_version = 'ada002'

    episodes_to_topics = {}
    episode_topics_sequence = []
    
    # get ordered list of all episodes
    response = fetch_simple_episodes(show_key)
    episodes = response['episodes']

    # for each episode:
    # - fetch all speakers ordered by scene_event count (how many lines they have)
    # - fetch all locations ordered by scene count
    # - transform results of both into lists of span dicts for creating plotly gantt charts
    episode_i = 0
    for episode in episodes:
        episode_key = episode['episode_key']
        episode_title = episode['title']
        episode_season = episode['season']
        sequence_in_season = episode['sequence_in_season']

        # fetch topics and scores
        response = episode_topic_vector_search(show_key, episode_key, topic_grouping, model_vendor, model_version)
        topics = response['topics']
        if len(topics) > topic_threshold:
            topics = topics[:topic_threshold]
        simple_topics = [dict(topic_key=t['topic_key'], breadcrumb=t['breadcrumb'], score=t['score']) for t in topics]
        episodes_to_topics[episode_key] = simple_topics
        # transform topics/scores to plotly-gantt-friendly span dicts
        for i in range(len(simple_topics)):
            topic_key = simple_topics[i]['topic_key']
            topic_cat = topic_key.split('.')[0]
            topic_span = dict(Task=topic_key, Start=episode_i, Finish=(episode_i+1), episode_key=episode_key, episode_title=episode_title, 
                              rank=i, topic_cat=topic_cat, season=episode_season, sequence_in_season=sequence_in_season,
                              info=f'{episode_title} (#{i+1} topic)')
            episode_topics_sequence.append(topic_span)

        episode_i += 1

    if overwrite_file:
        file_path = f'./app/data/topic_gantt_sequence_{show_key}_{topic_grouping}_{model_vendor}_{model_version}.csv'
        print(f'writing topic gantt sequence dataframe to file_path={file_path}')
        df = pd.DataFrame(episode_topics_sequence)
        df.to_csv(file_path)

    return {"episodes_to_topics": episodes_to_topics,
            "episode_topics_sequence": episode_topics_sequence}


@esr_app.get("/esr/generate_speaker_line_chart_sequences/{show_key}", tags=['ES Reader'])
def generate_speaker_line_chart_sequences(show_key: ShowKey, overwrite_file: bool = False):
    '''
    TODO 
    '''
    speaker_series_agg_word_counts = {spkr:0 for spkr in show_metadata[show_key.value]['regular_cast'].keys()}
    speaker_series_agg_line_counts = {spkr:0 for spkr in show_metadata[show_key.value]['regular_cast'].keys()}
    speaker_series_agg_scene_counts = {spkr:0 for spkr in show_metadata[show_key.value]['regular_cast'].keys()}
    speaker_series_agg_episode_counts = {spkr:0 for spkr in show_metadata[show_key.value]['regular_cast'].keys()}

    series_agg_word_count = 0
    series_agg_line_count = 0
    series_agg_scene_count = 0
    series_agg_episode_count = 0

    # get ordered list of all episodes
    response = fetch_simple_episodes(show_key)
    episodes = response['episodes']
    
    speaker_episode_rows = []
    episode_i = 0
    curr_season = None
    for episode in episodes:
        episode_key = str(episode['episode_key'])
        episode_title = episode['title']
        season = episode['season']
        sequence_in_season = episode['sequence_in_season']

        if not curr_season or season != curr_season:
            curr_season = season
            season_agg_word_count = 0
            season_agg_line_count = 0
            season_agg_scene_count = 0
            season_agg_episode_count = 0
            speaker_season_agg_word_counts = {spkr:0 for spkr in show_metadata[show_key.value]['regular_cast'].keys()}
            speaker_season_agg_line_counts = {spkr:0 for spkr in show_metadata[show_key.value]['regular_cast'].keys()}
            speaker_season_agg_scene_counts = {spkr:0 for spkr in show_metadata[show_key.value]['regular_cast'].keys()}
            speaker_season_agg_episode_counts = {spkr:0 for spkr in show_metadata[show_key.value]['regular_cast'].keys()}

        season_agg_episode_count += 1
        series_agg_episode_count += 1

        # fetch speakers and word counts
        word_count_agg_response = agg_dialog_word_counts(show_key, episode_key=episode_key)
        speaker_word_counts = word_count_agg_response['dialog_word_counts']
        episode_word_count = speaker_word_counts['_ALL_']
        season_agg_word_count += episode_word_count
        series_agg_word_count += episode_word_count
        # fetch speakers and line counts
        scene_event_agg_response = agg_scene_events_by_speaker(show_key, episode_key=episode_key)
        speaker_line_counts = scene_event_agg_response['scene_events_by_speaker']
        episode_line_count = speaker_line_counts['_ALL_']
        season_agg_line_count += episode_line_count
        series_agg_line_count += episode_line_count
        # fetch speakers and scene/episode counts
        scene_agg_response = agg_scenes_by_speaker(show_key, episode_key=episode_key)
        speaker_scene_counts = scene_agg_response['scenes_by_speaker']
        episode_scene_count = speaker_scene_counts['_ALL_']
        season_agg_scene_count += episode_scene_count
        series_agg_scene_count += episode_scene_count
        # episodes_to_speaker_counts[episode_key] = speaker_scene_counts.keys()

        for speaker in show_metadata[show_key.value]['regular_cast'].keys():
            if speaker in speaker_word_counts:
                # speaker_episode_row = {}
                word_count = speaker_word_counts[speaker] 
                line_count = speaker_line_counts[speaker]
                scene_count = speaker_scene_counts[speaker]
                # increment agg speaker counts
                speaker_season_agg_word_counts[speaker] += word_count
                speaker_series_agg_word_counts[speaker] += word_count
                speaker_season_agg_line_counts[speaker] += line_count
                speaker_series_agg_line_counts[speaker] += line_count
                speaker_season_agg_scene_counts[speaker] += scene_count
                speaker_series_agg_scene_counts[speaker] += scene_count
                speaker_season_agg_episode_counts[speaker] += 1
                speaker_series_agg_episode_counts[speaker] += 1
            else:
                word_count = 0 
                line_count = 0
                scene_count = 0

            # init speaker_episode_row
            speaker_episode_row = dict(
                speaker=speaker,
                episode_key=episode_key,
                episode_i=episode_i, 
                episode_title=episode_title,
                season=season,
                sequence_in_season=sequence_in_season,
                word_count=word_count, 
                line_count=line_count, 
                scene_count=scene_count)
            # speaker X counts as a % of episode X count
            speaker_episode_row['word_count_pct_of_episode'] = word_count / episode_word_count
            speaker_episode_row['line_count_pct_of_episode'] = line_count / episode_line_count
            speaker_episode_row['scene_count_pct_of_episode'] = scene_count / episode_scene_count
            # season agg speaker X counts as a % of season agg X count
            speaker_episode_row['word_count_pct_of_season'] = speaker_season_agg_word_counts[speaker] / season_agg_word_count
            speaker_episode_row['line_count_pct_of_season'] = speaker_season_agg_line_counts[speaker] / season_agg_line_count
            speaker_episode_row['scene_count_pct_of_season'] = speaker_season_agg_scene_counts[speaker] / season_agg_scene_count
            speaker_episode_row['episode_count_pct_of_season'] = speaker_season_agg_episode_counts[speaker] / season_agg_episode_count
            # overall agg speaker X counts as a % of overall agg X count
            speaker_episode_row['word_count_pct_of_series'] = speaker_series_agg_word_counts[speaker] / series_agg_word_count
            speaker_episode_row['line_count_pct_of_series'] = speaker_series_agg_line_counts[speaker] / series_agg_line_count
            speaker_episode_row['scene_count_pct_of_series'] = speaker_series_agg_scene_counts[speaker] / series_agg_scene_count
            speaker_episode_row['episode_count_pct_of_series'] = speaker_series_agg_episode_counts[speaker] / series_agg_episode_count
            
            speaker_episode_row['info'] = f'{speaker} in {episode_title}: {scene_count} scenes, {line_count} lines, {word_count} words'
            speaker_episode_rows.append(speaker_episode_row)

        episode_i += 1

    if overwrite_file:
        file_path = f'./app/data/speaker_episode_aggs_{show_key}.csv'
        print(f'writing speaker word/line/scene/episode counts and aggs dataframe to file_path={file_path}')
        df = pd.DataFrame(speaker_episode_rows)
        df.to_csv(file_path)

    return {"speaker_episode_rows": speaker_episode_rows}


@esr_app.get("/esr/generate_location_line_chart_sequences/{show_key}", tags=['ES Reader'])
def generate_location_line_chart_sequences(show_key: ShowKey, overwrite_file: bool = False):
    '''
    TODO 
    '''
    response = agg_scenes_by_location(show_key)
    locations = response['scenes_by_location']
    top_locations = [location for location, count in locations.items() if count > 10]
    location_series_agg_scene_counts = {location:0 for location in top_locations}
    location_series_agg_episode_counts = {location:0 for location in top_locations}

    series_agg_scene_count = 0
    series_agg_episode_count = 0

    # get ordered list of all episodes
    response = fetch_simple_episodes(show_key)
    episodes = response['episodes']
    
    location_episode_rows = []
    episode_i = 0
    curr_season = None
    for episode in episodes:
        episode_key = episode['episode_key']
        episode_title = episode['title']
        season = episode['season']
        sequence_in_season = episode['sequence_in_season']

        if not curr_season or season != curr_season:
            curr_season = season
            season_agg_scene_count = 0
            season_agg_episode_count = 0
            location_season_agg_scene_counts = {location:0 for location in top_locations}
            location_season_agg_episode_counts = {location:0 for location in top_locations}

        season_agg_episode_count += 1
        series_agg_episode_count += 1

        # fetch locations and scene/episode counts
        scene_agg_response = agg_scenes_by_location(show_key, episode_key=episode_key)
        location_scene_counts = scene_agg_response['scenes_by_location']
        episode_scene_count = location_scene_counts['_ALL_']
        del location_scene_counts['_ALL_']
        season_agg_scene_count += episode_scene_count
        series_agg_scene_count += episode_scene_count
        # episodes_to_speaker_counts[episode_key] = speaker_scene_counts.keys()

        for location in top_locations:
            if location in location_scene_counts:
                # location_episode_row = {}

                scene_count = location_scene_counts[location]
                # increment agg location counts
                location_season_agg_scene_counts[location] += scene_count
                location_series_agg_scene_counts[location] += scene_count
                location_season_agg_episode_counts[location] += 1
                location_series_agg_episode_counts[location] += 1
            else:
                scene_count = 0

            # init location_episode_row
            location_episode_row = dict(
                location=location,
                episode_i=episode_i, 
                episode_title=episode_title,
                season=season,
                sequence_in_season=sequence_in_season,
                scene_count=scene_count)
            # location scene counts as a % of episode scene count
            location_episode_row['scene_count_pct_of_episode'] = scene_count / episode_scene_count
            # season agg speaker scene/episode counts as a % of season agg scene/episode count
            location_episode_row['scene_count_pct_of_season'] = location_season_agg_scene_counts[location] / season_agg_scene_count
            location_episode_row['episode_count_pct_of_season'] = location_season_agg_episode_counts[location] / season_agg_episode_count
            # overall agg speaker scene/episode counts as a % of overall agg scene/episode count
            location_episode_row['scene_count_pct_of_series'] = location_series_agg_scene_counts[location] / series_agg_scene_count
            location_episode_row['episode_count_pct_of_series'] = location_series_agg_episode_counts[location] / series_agg_episode_count
            
            location_episode_row['info'] = f'{location} in {episode_title}: {scene_count} scenes'
            location_episode_rows.append(location_episode_row)

        episode_i += 1

    if overwrite_file:
        file_path = f'./app/data/location_episode_aggs_{show_key}.csv'
        print(f'writing location scene/episode counts and aggs dataframe to file_path={file_path}')
        df = pd.DataFrame(location_episode_rows)
        df.to_csv(file_path)

    return {"location_episode_rows": location_episode_rows}


# @esr_app.get("/esr/cluster_content/{show_key}/{num_clusters}", tags=['ES Reader'])
# def cluster_content(show_key: ShowKey, num_clusters: int, model_vendor: str = None, model_version: str = None):
#     if not model_vendor:
#         model_vendor = 'openai'
#     if not model_version:
#         model_version = 'ada002'

#     true_model_version = None

#     if model_vendor == 'openai':
#         vendor_meta = TRF_MODELS[model_vendor]
#         true_model_version = vendor_meta['versions'][model_version]['true_name']
#     else:
#         pass # TODO

#     vector_field = f'{model_vendor}_{model_version}_embeddings'
#     # fetch all model/vendor embeddings for show 
#     s = esqb.fetch_all_embeddings(show_key.value, vector_field)
#     es_query = s.to_dict()
#     doc_embeddings = esrt.return_all_embeddings(s, vector_field)
    
#     # cluster content
#     if num_clusters > len(doc_embeddings):
#         err = f'Unable to cluster {show_key} content: num_clusters={num_clusters} exceeds number of documents in corpus={len(doc_embeddings)}'
#         return {"error": err, "es_query": es_query}
#     doc_clusters, doc_clusters_df, _ = ef.cluster_docs(doc_embeddings, num_clusters)
#     # doc_clusters_df.set_index('doc_id').T.to_dict('list')
#     doc_clusters_df.to_dict('dict')

#     return {"doc_clusters": doc_clusters, "es_query": es_query}
