from fastapi import APIRouter
from operator import itemgetter

import nlp.embeddings_factory as ef
from nlp.nlp_metadata import WORD2VEC_VENDOR_VERSIONS as W2V_MODELS, TRANSFORMER_VENDOR_VERSIONS as TRF_MODELS
import nlp.query_preprocessor as qp
import es.es_query_builder as esqb
import es.es_response_transformer as esrt
from show_metadata import ShowKey


esr_app = APIRouter()



###################### ES FETCH ###########################

@esr_app.get("/esr/episode/{show_key}/{episode_key}", tags=['ES Reader'])
async def fetch_episode(show_key: ShowKey, episode_key: str, all_fields: bool = False):
    '''
    Fetch individual episode 
    '''
    s = await esqb.fetch_episode_by_key(show_key.value, episode_key, all_fields=all_fields)
    es_query = s.to_dict()
    match = await esrt.return_episode_by_key(s)
    return {"es_episode": match, 'es_query': es_query}


@esr_app.get("/esr/search_doc_ids/{show_key}", tags=['ES Reader'])
def search_doc_ids(show_key: ShowKey, season: str = None):
    '''
    Get all es source _ids for show 
    '''
    s = esqb.search_doc_ids(show_key.value, season=season)
    es_query = s.to_dict()
    matches = esrt.return_doc_ids(s)
    return {"doc_count": len(matches), "doc_ids": matches, "es_query": es_query}


@esr_app.get("/esr/list_episodes_by_season/{show_key}", tags=['ES Reader'])
def list_episodes_by_season(show_key: ShowKey):
    '''
    Fetch simple (sceneless) episodes sequenced and grouped by season
    '''
    s = esqb.list_episodes_by_season(show_key.value)
    es_query = s.to_dict()
    episodes_by_season = esrt.return_episodes_by_season(s)
    return {"episodes_by_season": episodes_by_season, "es_query": es_query}



###################### ES SEARCH ###########################

@esr_app.get("/esr/search_episodes_by_title/{show_key}", tags=['ES Reader'])
async def search_episodes_by_title(show_key: ShowKey, title: str = None):
    '''
    Free text episode search by title
    '''
    s = await esqb.search_episodes_by_title(show_key.value, title)
    es_query = s.to_dict()
    matches = await esrt.return_episodes_by_title(s)
    return {"episode_count": len(matches), "episodes": matches, "es_query": es_query}


@esr_app.get("/esr/search_scenes/{show_key}", tags=['ES Reader'])
async def search_scenes(show_key: ShowKey, season: str = None, episode_key: str = None, location: str = None, description: str = None):
    '''
    Facet query of nested Scene fields 
    '''
    if not (location or description):
        error = 'Unable to execute search_scenes without at least one scene property set (location or description)'
        print(error)
        return {"error": error}
    s = await esqb.search_scenes(show_key.value, season=season, episode_key=episode_key, location=location, description=description)
    es_query = s.to_dict()
    matches, scene_count = await esrt.return_scenes(s)
    return {"episode_count": len(matches), "scene_count": scene_count, "matches": matches, "es_query": es_query}


@esr_app.get("/esr/search_scene_events/{show_key}", tags=['ES Reader'])
async def search_scene_events(show_key: ShowKey, season: str = None, episode_key: str = None, speaker: str = None, dialog: str = None, location: str = None):
    '''
    Facet query of nested Scene and SceneEvent fields 
    '''
    s = await esqb.search_scene_events(show_key.value, season=season, episode_key=episode_key, speaker=speaker, dialog=dialog)
    es_query = s.to_dict()
    matches, scene_count, scene_event_count = await esrt.return_scene_events(s, location=location)
    return {"episode_count": len(matches), "scene_count": scene_count, "scene_event_count": scene_event_count, "matches": matches, "es_query": es_query}


@esr_app.get("/esr/search_scene_events_multi_speaker/{show_key}", tags=['ES Reader'])
async def search_scene_events_multi_speaker(show_key: ShowKey, season: str = None, episode_key: str = None, speakers: str = None, location: str = None):
    '''
    Facet query of Scenes comprised of SceneEvents matching 1-n speakers
    '''
    s = await esqb.search_scene_events_multi_speaker(show_key.value, speakers, season=season, episode_key=episode_key)
    es_query = s.to_dict()
    matches, scene_count, scene_event_count = await esrt.return_scene_events_multi_speaker(s, speakers, location=location)
    return {"episode_count": len(matches), "scene_count": scene_count, "scene_event_count": scene_event_count, "matches": matches, "es_query": es_query}


@esr_app.get("/esr/search/{show_key}", tags=['ES Reader'])
async def search(show_key: ShowKey, season: str = None, episode_key: str = None, qt: str = None):
    '''
    Generic free text search of Episodes and nested Scenes and SceneEvents
    '''
    s = await esqb.search_episodes(show_key.value, season=season, episode_key=episode_key, qt=qt)
    es_query = s.to_dict()
    matches, scene_count, scene_event_count = await esrt.return_episodes(s)
    return {"episode_count": len(matches), "scene_count": scene_count, "scene_event_count": scene_event_count, "matches": matches, "es_query": es_query}


# TODO support POST for long requests?
@esr_app.get("/esr/vector_search/{show_key}", tags=['ES Reader'])
def vector_search(show_key: ShowKey, qt: str, model_vendor: str = None, model_version: str = None, season: str = None):
    '''
    Generates vector embedding for qt, then determines vector cosine similarity to indexed documents using k-nearest neighbors search
    '''
    if not model_vendor:
        model_vendor = 'webvectors'
    if not model_version:
        model_version = '223'

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


@esr_app.get("/esr/test_vector_search/{show_key}", tags=['ES Reader'])
def test_vector_search(show_key: ShowKey, qt: str, model_vendor: str = None, model_version: str = None, normalize_and_expand: bool = False):
    '''
    Experimental endpoint for troubleshooting ontology overrides and other qt alterations preceding vectorization
    '''
    if not model_vendor:
        model_vendor = 'webvectors'
    if not model_version:
        model_version = '223'

    # NOTE currently only set up for word2vec, not for openai embeddings

    vendor_meta = W2V_MODELS[model_vendor]
    tag_pos = vendor_meta['pos_tag']

    try:
        if normalize_and_expand:
            qt = qp.normalize_and_expand_query_vocab(qt, show_key)
        tokenized_qt = qp.tokenize_and_remove_stopwords(qt, tag_pos=tag_pos)
    except Exception as e:
        return {"error": e}
    return {"normd_expanded_qt": qt, "tokenized_qt": tokenized_qt}



###################### ES AGGREGATIONS ###########################

@esr_app.get("/esr/agg_episodes/{show_key}", tags=['ES Reader'])
async def agg_episodes(show_key: ShowKey, season: str = None, location: str = None):
    s = await esqb.agg_episodes(show_key.value, season=season, location=location)
    es_query = s.to_dict()
    episode_count = await esrt.return_episode_count(s)
    return {"episode_count": episode_count, "es_query": es_query}


@esr_app.get("/esr/agg_episodes_by_speaker/{show_key}", tags=['ES Reader'])
async def agg_episodes_by_speaker(show_key: ShowKey, season: str = None, location: str = None, other_speaker: str = None):
    s = await esqb.agg_episodes_by_speaker(show_key.value, season=season, location=location, other_speaker=other_speaker)
    es_query = s.to_dict()
    # separate call to get episode_count without double-counting per speaker
    episode_count = await agg_episodes(show_key, season=season, location=location)
    matches = await esrt.return_episodes_by_speaker(s, episode_count['episode_count'], location=location, other_speaker=other_speaker)
    return {"speaker_count": len(matches), "episodes_by_speaker": matches, "es_query": es_query}


@esr_app.get("/esr/agg_scenes_by_location/{show_key}", tags=['ES Reader'])
async def agg_scenes_by_location(show_key: ShowKey, season: str = None, episode_key: str = None, speaker: str = None):
    s = await esqb.agg_scenes_by_location(show_key.value, season=season, episode_key=episode_key, speaker=speaker)
    es_query = s.to_dict()
    matches = await esrt.return_scenes_by_location(s, speaker=speaker)
    return {"location_count": len(matches), "scenes_by_location": matches, "es_query": es_query}


@esr_app.get("/esr/agg_scenes_by_speaker/{show_key}", tags=['ES Reader'])
async def agg_scenes_by_speaker(show_key: ShowKey, season: str = None, episode_key: str = None, location: str = None, other_speaker: str = None):
    s = await esqb.agg_scenes_by_speaker(show_key.value, season=season, episode_key=episode_key, location=location, other_speaker=other_speaker)
    es_query = s.to_dict()
    # separate call to get scene_count without double-counting per speaker
    scene_count = await agg_scenes(show_key, season=season, episode_key=episode_key, location=location)
    matches = await esrt.return_scenes_by_speaker(s, scene_count['scene_count'], location=location, other_speaker=other_speaker)
    return {"speaker_count": len(matches), "scenes_by_speaker": matches, "es_query": es_query}


@esr_app.get("/esr/agg_scenes/{show_key}", tags=['ES Reader'])
async def agg_scenes(show_key: ShowKey, season: str = None, episode_key: str = None, location: str = None):
    s = await esqb.agg_scenes(show_key.value, season=season, episode_key=episode_key, location=location)
    es_query = s.to_dict()
    scene_count = await esrt.return_scene_count(s)
    return {"scene_count": scene_count, "es_query": es_query}


@esr_app.get("/esr/agg_scene_events_by_speaker/{show_key}", tags=['ES Reader'])
async def agg_scene_events_by_speaker(show_key: ShowKey, season: str = None, episode_key: str = None, dialog: str = None):
    s = await esqb.agg_scene_events_by_speaker(show_key.value, season=season, episode_key=episode_key, dialog=dialog)
    es_query = s.to_dict()
    matches = await esrt.return_scene_events_by_speaker(s, dialog=dialog)
    return {"speaker_count": len(matches), "scene_events_by_speaker": matches, "es_query": es_query}


@esr_app.get("/esr/agg_dialog_word_counts/{show_key}", tags=['ES Reader'])
async def agg_dialog_word_counts(show_key: ShowKey, season: str = None, episode_key: str = None, speaker: str = None):
    s = await esqb.agg_dialog_word_counts(show_key.value, season=season, episode_key=episode_key, speaker=speaker)
    es_query = s.to_dict()
    matches = await esrt.return_dialog_word_counts(s, speaker=speaker)
    return {"dialog_word_counts": matches, "es_query": es_query}


@esr_app.get("/esr/composite_speaker_aggs/{show_key}", tags=['ES Reader'])
async def composite_speaker_aggs(show_key: ShowKey, season: str = None, episode_key: str = None):
    if not episode_key:
        speaker_episode_counts = await agg_episodes_by_speaker(show_key, season=season)
    speaker_scene_counts = await agg_scenes_by_speaker(show_key, season=season, episode_key=episode_key)
    speaker_line_counts = await agg_scene_events_by_speaker(show_key, season=season, episode_key=episode_key)
    speaker_word_counts = await agg_dialog_word_counts(show_key, season=season, episode_key=episode_key)

    # TODO refactor this to generically handle dicts threading together
    speakers = {}
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



###################### OTHER ES READS ###########################

@esr_app.get("/esr/keywords_by_episode/{show_key}/{episode_key}", tags=['ES Reader'])
async def keywords_by_episode(show_key: ShowKey, episode_key: str, exclude_speakers: bool = False):
    response = await esqb.keywords_by_episode(show_key.value, episode_key)
    all_speakers = []
    if exclude_speakers:
        res = await agg_scenes_by_speaker(show_key, episode_key=episode_key) # TODO should this use agg_episodes_by_speaker now?
        all_speakers = res['scenes_by_speaker'].keys()
    matches = await esrt.return_keywords_by_episode(response, exclude_terms=all_speakers)
    return {"keyword_count": len(matches), "keywords": matches}


@esr_app.get("/esr/keywords_by_corpus/{show_key}", tags=['ES Reader'])
async def keywords_by_corpus(show_key: ShowKey, season: str = None, exclude_speakers: bool = False):
    response = await esqb.keywords_by_corpus(show_key.value, season=season)
    all_speakers = []
    if exclude_speakers:
        res = await agg_episodes_by_speaker(show_key, season=season)
        all_speakers = res['episodes_by_speaker'].keys()
    matches = await esrt.return_keywords_by_corpus(response, exclude_terms=all_speakers)
    return {"keyword_count": len(matches), "keywords": matches}


@esr_app.get("/esr/more_like_this/{show_key}/{episode_key}", tags=['ES Reader'])
async def more_like_this(show_key: ShowKey, episode_key: str):
    s = await esqb.more_like_this(show_key.value, episode_key)
    es_query = s.to_dict()
    matches = await esrt.return_more_like_this(s)
    return {"similar_episode_count": len(matches), "similar_episodes": matches, "es_query": es_query}


@esr_app.get("/esr/cluster_content/{show_key}/{num_clusters}", tags=['ES Reader'])
def cluster_content(show_key: ShowKey, num_clusters: int, model_vendor: str = None, model_version: str = None):
    if not model_vendor:
        model_vendor = 'openai'
    if not model_version:
        model_version = 'ada002'

    true_model_version = None

    if model_vendor == 'openai':
        vendor_meta = TRF_MODELS[model_vendor]
        true_model_version = vendor_meta['versions'][model_version]['true_name']
    else:
        pass # TODO

    vector_field = f'{model_vendor}_{model_version}_embeddings'
    # fetch all model/vendor embeddings for show 
    s = esqb.fetch_all_embeddings(show_key.value, vector_field)
    es_query = s.to_dict()
    doc_embeddings = esrt.return_all_embeddings(s, vector_field)
    
    # cluster content
    if num_clusters > len(doc_embeddings):
        err = f'Unable to cluster {show_key} content: num_clusters={num_clusters} exceeds number of documents in corpus={len(doc_embeddings)}'
        return {"error": err, "es_query": es_query}
    doc_clusters, doc_clusters_df, embeddings_matrix = ef.cluster_docs(doc_embeddings, num_clusters)
    # doc_clusters_df.set_index('doc_id').T.to_dict('list')
    doc_clusters_df.to_dict('dict')

    return {"doc_clusters": doc_clusters, "es_query": es_query}
