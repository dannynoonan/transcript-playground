# from datetime import datetime
# from elasticsearch import Elasticsearch
# from elasticsearch import RequestsHttpConnection
from elasticsearch_dsl import Search, connections, Q, A
from elasticsearch_dsl.query import MoreLikeThis

from app.config import settings
from app.es.es_metadata import STOPWORDS, VECTOR_FIELDS, RELATIONS_FIELDS
from app.es.es_model import EsEpisodeTranscript, EsTopic, EsSpeaker, EsSpeakerSeason, EsSpeakerEpisode
import app.es.es_read_router as esr
from app.show_metadata import ShowKey


# es_client = Elasticsearch(
#     hosts=[{'host': settings.es_host, 'port': settings.es_port, 'scheme': 'https'}],    
#     basic_auth=(settings.es_user, settings.es_password),
#     verify_certs=False
#     # connection_class=RequestsHttpConnection
# )

# connections.create_connection(hosts=['http://localhost:9200'], timeout=20)

es_conn = connections.create_connection(hosts=[{'host': settings.es_host, 'port': settings.es_port, 'scheme': 'https'}],
                                        basic_auth=(settings.es_user, settings.es_password), verify_certs=False, timeout=20)

# connections.configure(
#     default={'hosts': 'http://localhost:9200'},
    # dev={
    #     'hosts': ['http://localhost:9200'],
    #     'sniff_on_start': True
    # }
# )


def init_transcripts_index():
    # EsEpisodeTranscript.init(using=es_client)
    EsEpisodeTranscript.init()
    es_conn.indices.put_settings(index="transcripts", body={"index": {"max_inner_result_window": 1000}})


def init_topics_index():
    EsTopic.init()
    es_conn.indices.put_settings(index="topics", body={"index": {"max_inner_result_window": 1000}})


def init_speakers_index():
    EsSpeaker.init()
    es_conn.indices.put_settings(index="speakers", body={"index": {"max_inner_result_window": 1000}})


def init_speaker_seasons_index():
    EsSpeakerSeason.init()
    es_conn.indices.put_settings(index="speaker_seasons", body={"index": {"max_inner_result_window": 1000}})


def init_speaker_episodes_index():
    EsSpeakerEpisode.init()
    es_conn.indices.put_settings(index="speaker_episodes", body={"index": {"max_inner_result_window": 1000}})


def save_es_episode(es_episode: EsEpisodeTranscript) -> None:
    # es_episode.save(using=es_client)
    es_episode.save()
    # persisted_es_episode = EsEpisodeTranscript.get(id=es_episode.meta.id, ignore=404)
    # if persisted_es_episode:
    #     es_episode.update(using=es, doc_as_upsert=True)
    # else:
    #     es_episode.save(using=es)


def save_es_topic(es_topic: EsTopic) -> None:
    # TODO this is functionally identical to `save_es_episode`, do we even need either of them?
    es_topic.save()


def save_es_speaker(es_speaker: EsSpeaker) -> None:
    # TODO this is functionally identical to `save_es_episode` and `save_es_topic`, do we even need any of them?
    es_speaker.save()


def save_es_speaker_season(es_speaker_season: EsSpeakerSeason) -> None:
    # TODO this is functionally identical to other es `save` functions 
    es_speaker_season.save()


def save_es_speaker_episode(es_speaker_episode: EsSpeakerEpisode) -> None:
    # TODO this is functionally identical to other es `save` functions 
    es_speaker_episode.save()


def fetch_episode_by_key(show_key: str, episode_key: str, all_fields: bool = False) -> Search:
    print(f'begin fetch_episode_by_key for show_key={show_key} episode_key={episode_key}')

    # s = Search(using=es_client, index='transcripts')
    s = Search(index='transcripts')
    s = s.extra(size=1)

    s = s.filter('term', show_key=show_key)
    s = s.filter('term', episode_key=episode_key)
    if not all_fields:
        s = s.source(excludes=['flattened_text'] + VECTOR_FIELDS + RELATIONS_FIELDS)

    return s


def fetch_doc_ids(show_key: str, season: str = None) -> Search:
    print(f'begin fetch_doc_ids for show_key={show_key} season={season}')

    s = Search(index='transcripts')
    s = s.extra(size=1000)
    s = s.extra(stored_fields=['_id'])

    s = s.filter('term', show_key=show_key)
    if season:
        s = s.filter('term', season=season)
    
    return s


# NOTE I'm not sure what this was for, probably created during BERTopic experimentation
# def fetch_flattened_episodes(show_key: str, season: str = None) -> Search:
#     print(f'begin fetch_flattened_episodes for show_key={show_key} season={season}')

#     s = Search(index='transcripts')
#     s = s.extra(size=1000)

#     s = s.filter('term', show_key=show_key)
#     if season:
#         s = s.filter('term', season=season)

#     s = s.sort('season', 'sequence_in_season')

#     s = s.source(excludes=['scenes'] + VECTOR_FIELDS + RELATIONS_FIELDS)

#     return s


def fetch_simple_episodes(show_key: str, season: str = None) -> Search:
    print(f'begin fetch_simple_episodes for show_key={show_key} season={season}')

    s = Search(index='transcripts')
    s = s.extra(size=1000)

    s = s.filter('term', show_key=show_key)
    if season:
        s = s.filter('term', season=season)

    s = s.sort('season', 'sequence_in_season')

    s = s.source(excludes=['flattened_text', 'scenes'] + VECTOR_FIELDS + RELATIONS_FIELDS)

    return s


def search_episodes_by_title(show_key: str, qt: str) -> Search:
    print(f'begin search_episodes_by_title for show_key={show_key} qt={qt}')

    s = Search(index='transcripts')
    s = s.extra(size=1000)

    q = Q('bool', must=[Q('match', title=qt)])

    s = s.filter('term', show_key=show_key)
    s = s.query(q)
    s = s.highlight('title')

    return s


def fetch_speaker(show_key: str, speaker_name: str) -> EsSpeaker|None:
    print(f'begin fetch_speaker for show_key={show_key} speaker_name={speaker_name}')

    doc_id = f'{show_key}_{speaker_name}'

    try:
        speaker = EsSpeaker.get(id=doc_id, index='speakers')
    except Exception as e:
        print(f'Failed to fetch speaker `{speaker_name}` for show_key=`{show_key}`')
        return None
    
    # TODO don't like having to do this 
    if speaker.child_topics:
        speaker.child_topics = speaker.child_topics._d_
    if speaker.parent_topics:
        speaker.parent_topics = speaker.parent_topics._d_

    return speaker


def fetch_indexed_speakers(show_key: str, return_fields: list = []) -> Search:
    print(f'begin fetch_indexed_speakers for show_key={show_key}')

    s = Search(index='speakers')
    s = s.extra(size=1000)

    s = s.filter('term', show_key=show_key)

    s = s.sort({"episode_count" : {"order" : "desc"}}, {"scene_count" : {"order" : "desc"}})

    if return_fields:
        s = s.source(includes=return_fields)

    return s


# def fetch_speaker_embedding(show_key: str, speaker: str, vector_field: str) -> Search:
#     print(f'begin fetch_speaker_embedding for show_key={show_key} speaker={speaker} vector_field={vector_field}')

#     s = Search(index='speakers')
#     s = s.extra(size=1)

#     s = s.filter('term', show_key=show_key)
#     s = s.filter('term', speaker=speaker)
#     s = s.source(includes=[vector_field])

#     return s


def fetch_speaker_season(show_key: str, speaker_name: str, season: int) -> EsSpeakerSeason|None:
    print(f'begin fetch_speaker_season for show_key={show_key} speaker_name={speaker_name} season={season}')

    doc_id = f'{show_key}_{speaker_name}_{season}'

    try:
        speaker_season = EsSpeakerSeason.get(id=doc_id, index='speaker_seasons')
    except Exception as e:
        print(f'Failed to fetch speaker_season for show_key={show_key} speaker={speaker_name} season={season}', e)
        return None

    return speaker_season


def fetch_speaker_seasons(show_key: str, speaker: str, season: int = None, return_fields: list = []) -> Search:
    print(f'begin fetch_speaker_seasons for show_key={show_key} speaker={speaker} season={season} return_fields={return_fields}')

    s = Search(index='speaker_seasons')
    s = s.extra(size=1000)

    s = s.filter('term', show_key=show_key)
    s = s.filter('term', speaker=speaker)
    if season:
        s = s.filter('term', season=season)

    s = s.sort('season')

    if return_fields:
        s = s.source(includes=return_fields)

    return s


# def fetch_speaker_season_embeddings(show_key: str, speaker: str, vector_field: str, season: int = None) -> Search:
#     print(f'begin fetch_speaker_season_embeddings for show_key={show_key} speaker={speaker} vector_field={vector_field}')

#     s = Search(index='speaker_seasons')
#     s = s.extra(size=1000)

#     s = s.filter('term', show_key=show_key)
#     s = s.filter('term', speaker=speaker)
#     if season:
#         s = s.filter('term', season=season)

#     s = s.sort('season')

#     s = s.source(includes=[vector_field])

#     return s


def fetch_speaker_episode(show_key: str, speaker_name: str, episode_key: str) -> EsSpeakerEpisode|None:
    print(f'begin fetch_speaker_episode for show_key={show_key} speaker_name={speaker_name} episode_key={episode_key}')

    doc_id = f'{show_key}_{speaker_name}_{episode_key}'

    try:
        speaker_episode = EsSpeakerEpisode.get(id=doc_id, index='speaker_episodes')
    except Exception as e:
        print(f'Failed to fetch speaker_episode for show_key={show_key} speaker={speaker_name} episode_key={episode_key}', e)
        return None

    return speaker_episode


def fetch_speaker_episodes(show_key: str, speaker: str, season: int = None, episode_key: str = None, return_fields: list = []) -> Search:
    print(f'begin fetch_speaker_episodes for show_key={show_key} speaker={speaker} episode_key={episode_key} season={season} return_fields={return_fields}')

    s = Search(index='speaker_episodes')
    s = s.extra(size=1000)

    s = s.filter('term', show_key=show_key)
    s = s.filter('term', speaker=speaker)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    elif season:
        s = s.filter('term', season=season)

    # s = s.sort('agg_score', order='desc')
    s = s.sort({"agg_score" : {"order" : "desc"}})

    if return_fields:
        s = s.source(includes=return_fields)

    return s


# def fetch_speaker_episode_embeddings(show_key: str, speaker: str, vector_field: str, season: int = None, episode_key: str = None) -> Search:
#     print(f'begin fetch_speaker_episode_embeddings for show_key={show_key} speaker={speaker} vector_field={vector_field}')

#     s = Search(index='speaker_episodes')
#     s = s.extra(size=1000)

#     s = s.filter('term', show_key=show_key)
#     s = s.filter('term', speaker=speaker)
#     if episode_key:
#         s = s.filter('term', episode_key=episode_key)
#     elif season:
#         s = s.filter('term', season=season)
#     s = s.source(includes=[vector_field])

#     return s


def fetch_speaker_embeddings(show_key: str, speaker: str, vector_field: str, seasons: list = [], episode_keys: list = [], min_depth: bool = True) -> tuple[list, dict, dict]:
    print(f'begin fetch_speaker_embeddings for show_key={show_key} speaker={speaker} vector_field={vector_field} seasons={seasons} episode_keys={episode_keys}')

    try:
        es_speaker = fetch_speaker(show_key, speaker)
        speaker_series_embeddings = getattr(es_speaker, vector_field)
    except Exception as e:
        return {"error": f"Failed fetch_speaker for show_key={show_key} speaker={speaker}: {e}"}
    
    if min_depth and speaker_series_embeddings:
         return speaker_series_embeddings, {}, {}
            
    if not seasons:
        seasons = es_speaker.seasons_to_episode_keys._d_.keys()
    if not episode_keys:
        for s, e_keys in es_speaker.seasons_to_episode_keys._d_.items():
            if s in seasons:
                episode_keys.extend(e_keys)

    season_embeddings = {}
    for season in seasons:
        try:
            es_speaker_season = fetch_speaker_season(show_key, speaker, season)
            if not es_speaker_season:
                print(f"Failed fetch_speaker_episode for show_key={show_key} speaker={speaker} season={season}")
                continue
            speaker_season_embeddings = getattr(es_speaker_season, vector_field)
            if speaker_season_embeddings:
                season_embeddings[season] = speaker_season_embeddings
        except Exception as e:
            print(f"Failed fetch_speaker_season for show_key={show_key} speaker={speaker} season={season}: {e}")

    if min_depth and len(speaker_season_embeddings) == len(seasons):
        return speaker_series_embeddings, season_embeddings, {}
    
    episode_embeddings = {}

    for episode_key in episode_keys:
        try:
            es_speaker_episode = fetch_speaker_episode(show_key, speaker, episode_key)
            if not es_speaker_episode:
                print(f"Failed fetch_speaker_episode for show_key={show_key} speaker={speaker} episode_key={episode_key}")
                continue
            speaker_episode_embeddings = getattr(es_speaker_episode, vector_field)
            if speaker_episode_embeddings:
                episode_embeddings[episode_key] = speaker_episode_embeddings
        except Exception as e:
            print(f"Failed fetch_speaker_episode for show_key={show_key} speaker={speaker} episode_key={episode_key}: {e}")

    return speaker_series_embeddings, season_embeddings, episode_embeddings


def fetch_topic(topic_grouping: str, topic_key: str) -> EsTopic|None:
    print(f'begin fetch_topic for topic_grouping={topic_grouping} topic_key={topic_key}')

    doc_id = f'{topic_grouping}_{topic_key}'

    try:
        topic = EsTopic.get(id=doc_id, index='topics')
    except Exception as e:
        print(f'Failed to fetch topic with topic_grouping={topic_grouping} topic_key={topic_key}')
        return None

    return topic


def fetch_topic_grouping(topic_grouping: str, return_fields: list = None) -> Search:
    print(f'begin fetch_topics for topic_grouping={topic_grouping}')

    s = Search(index='topics')
    s = s.extra(size=1000)

    s = s.filter('term', topic_grouping=topic_grouping)

    s = s.sort('parent_key', 'topic_key')

    if return_fields:
        s = s.source(includes=return_fields)

    return s


def search_speakers_by_topic(topic_grouping: str, topic_key: str, is_parent: bool = False, show_key: str = None, min_word_count: int = None) -> Search:
    print(f'begin search_speakers_by_topic for topic_grouping={topic_grouping} topic_key={topic_key} is_parent={is_parent} show_key={show_key} min_word_count={min_word_count}')

    s = Search(index='speakers')
    s = s.extra(size=1000)

    if is_parent:
        topic_path = f'parent_topics.{topic_grouping}'
    else:
        topic_path = f'child_topics.{topic_grouping}'
    topic_key_path = f'{topic_path}.topic_key'

    s = s.filter('match', **{topic_key_path: topic_key})
    if show_key:
        s = s.filter('term', show_key=show_key)
    if min_word_count:
        s = s.filter('range', word_count={'gt': min_word_count})

    # TODO this doesn't work like you want it to, need to nest topics as InnerDocs
    topic_score_path = f'{topic_path}.score'
    s = s.sort(topic_score_path)

    s = s.source(excludes=['lines', 'seasons_to_episode_keys', 'openai_ada002_embeddings'])

    return s


def search_scenes(show_key: str, season: str = None, episode_key: str = None, location: str = None, description: str = None) -> Search:
    print(f'begin search_scenes for show_key={show_key} season={season} episode_key={episode_key} location={location} description={description}')

    if not (location or description):
        print(f'Warning: unable to execute search_scenes without at least one scene_event property set (location or description)')
        return None
    
    s = Search(index='transcripts')
    s = s.extra(size=1000)

    # q = Q("match", scenes__location=qt)
    location_q = Q('match', **{'scenes.location': location})
    description_q = Q('match', **{'scenes.description': description})

    q = None
    if location:
        q = location_q
        if description:
            q = q & description_q
    else:
        q = description_q

    s = s.query('nested', path='scenes', 
            query=q,
            inner_hits={
                'size': 1000, 
                'highlight': {
                    'fields': {
                        'scenes.location': {}, 
                        'scenes.description': {}
                    }
                }
            })

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    s = s.source(excludes=['flattened_text', 'scenes'] + VECTOR_FIELDS + RELATIONS_FIELDS)

    return s


def search_scene_events(show_key: str, season: str = None, episode_key: str = None, speaker: str = None, dialog: str = None) -> Search:
    print(f'begin search_scene_events for show_key={show_key} season={season} episode_key={episode_key} speaker={speaker} dialog={dialog}')
    
    if not (speaker or dialog):
        print(f'Warning: unable to execute search_scene_events without at least one scene_event property set (speaker or dialog)')
        return []

    s = Search(index='transcripts')
    s = s.extra(size=1000)

    speaker_q = Q('match', **{'scenes.scene_events.spoken_by.keyword': speaker})
    dialog_q = Q('match', **{'scenes.scene_events.dialog': dialog})

    q = None
    if speaker:
        q = speaker_q
        if dialog:
            q = q & dialog_q
    else:
        q = dialog_q

    nested_q = Q('nested', path='scenes.scene_events', 
            query=q,
            inner_hits={
                'size': 1000, 
                'highlight': {
                    'fields': {
                        'scenes.scene_events.spoken_by': {}, 
                        'scenes.scene_events.dialog': {}
                    }
                }
            })
    # if location:
    #     nested_q = nested_q & Q('nested', path='scenes', 
    #         query=Q('match', **{'scenes.location': location}),
    #         inner_hits={'size': 1000})

    s = s.query(nested_q)

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)
    # if location:
    #     s = s.filter('nested', path='scenes', query=Q('match', **{'scenes.location': location}))

    s = s.source(excludes=['flattened_text', 'scenes.scene_events'] + VECTOR_FIELDS + RELATIONS_FIELDS)

    return s


def search_scene_events_multi_speaker(show_key: str, speakers: str, season: str = None, episode_key: str = None) -> Search:
    print(f'begin search_scene_events_multi_speaker for show_key={show_key} season={season} episode_key={episode_key} speakers={speakers}')

    speakers = speakers.split(',')

    s = Search(index='transcripts')
    s = s.extra(size=1000)

    must_nested_qs = None

    for speaker in speakers: 
        speaker_q = Q('bool', must=[Q('match', **{'scenes.scene_events.spoken_by.keyword': speaker})])

        nested_q = Q('nested', path='scenes.scene_events', 
                query=speaker_q,
                inner_hits={
                    'name': speaker,
                    'size': 1000, 
                    'highlight': {
                        'fields': {
                            'scenes.scene_events.spoken_by': {}, 
                            'scenes.scene_events.dialog': {}
                        }
                    }
                })
        if not must_nested_qs:
            must_nested_qs = nested_q
        else:
            must_nested_qs = must_nested_qs & nested_q

    s = s.query(must_nested_qs)

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    s = s.source(excludes=['flattened_text', 'scenes.scene_events'] + VECTOR_FIELDS + RELATIONS_FIELDS)

    return s


def search_episodes(show_key: str, season: str = None, episode_key: str = None, qt: str = None) -> Search:
    print(f'begin search for show_key={show_key} season={season} episode_key={episode_key} qt={qt}')

    '''
    qt can be: 
        title
        scenes.location
        scenes.description
        scenes.scene_events.context_info
        scenes.scene_events.spoken_by
        scenes.scene_events.dialog
    '''

    s = Search(index='transcripts')
    s = s.extra(size=1000)

    episode_q = Q('match', title=qt)
    
    scene_fields_q = Q('bool', minimum_should_match=1, should=[
            Q('match', **{'scenes.location': qt}),
            Q('match', **{'scenes.description': qt})])
    
    scenes_q = Q('nested', path='scenes', 
            query=scene_fields_q,
            inner_hits={
                'size': 1000, 
                'highlight': {
                    'fields': {
                        'scenes.location': {}, 
                        'scenes.description': {}
                    }
                }
            })
    
    scene_event_fields_q = Q('bool', minimum_should_match=1, should=[
            Q('match', **{'scenes.scene_events.context_info': qt}),
            Q('match', **{'scenes.scene_events.spoken_by.keyword': qt}),
            Q('match', **{'scenes.scene_events.dialog': qt})])
    
    scene_events_q = Q('nested', path='scenes.scene_events', 
            query=scene_event_fields_q,
            inner_hits={
                'size': 1000, 
                'highlight': {
                    'fields': {
                        'scenes.scene_events.context_info': {},
                        'scenes.scene_events.spoken_by': {}, 
                        'scenes.scene_events.dialog': {}
                    }
                }
            })

    s = s.query(episode_q | scenes_q | scene_events_q)

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    s = s.highlight('title')

    s = s.source(excludes=['flattened_text', 'scenes.scene_events'] + VECTOR_FIELDS + RELATIONS_FIELDS)

    return s


def fetch_all_episode_relations(show_key: str, model_vendor: str, model_version: str) -> Search:
    print(f'begin fetch_all_episode_relations for show_key={show_key} model_vendor={model_vendor} model_version={model_version}')

    s = Search(index='transcripts')
    s = s.extra(size=1000)

    s = s.filter('term', show_key=show_key)

    s = s.sort('season', 'sequence_in_season')

    relations_field = f'{model_vendor}_{model_version}_relations_text'

    # s = s.source(excludes=['flattened_text', 'scenes'] + VECTOR_FIELDS)
    s = s.source(includes=['episode_key', 'title', 'season', relations_field])

    return s


def agg_seasons(show_key: str, location: str = None) -> Search:
    print(f'begin agg_episodes for show_key={show_key} location={location}')

    s = Search(index='transcripts')
    s = s.extra(size=0)

    s = s.filter('term', show_key=show_key)

    s.aggs.bucket('by_season', 'terms', field='season', size=1000)

    # TODO location

    return s


def agg_episodes(show_key: str, season: str = None, location: str = None) -> Search:
    print(f'begin agg_episodes for show_key={show_key} season={season} location={location}')

    s = Search(index='transcripts')
    s = s.extra(size=0)

    s = s.filter('term', show_key=show_key)
    if season:
        s = s.filter('term', season=season)

    # TODO location

    return s


def agg_seasons_by_speaker(show_key: str, location: str = None) -> Search:
    print(f'begin agg_episodes_by_speaker for show_key={show_key} location={location}')

    s = Search(index='transcripts')
    s = s.extra(size=0)

    s = s.filter('term', show_key=show_key)

    if location:
        pass  # TODO copied from agg_episodes_by_speaker
        # s.aggs.bucket(
        #     'scenes', 'nested', path='scenes'
        # ).bucket(
        #     'location_match', 'filter', filter={"match": {"scenes.location": location}}
        # ).bucket(
        #     'scene_events', 'nested', path='scenes.scene_events'
        # ).bucket(
        #     'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=1000
        # ).bucket(
        #     'for_episode', 'reverse_nested'
        # )
    else:
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=2000
        ).bucket(
            'by_season', 'reverse_nested'
        ).bucket(
            'season', 'terms', field='season'
        )
    
    return s


def agg_seasons_by_location(show_key: str) -> Search:
    print(f'begin agg_episodes_by_speaker for show_key={show_key}')

    s = Search(index='transcripts')
    s = s.extra(size=0)

    s = s.filter('term', show_key=show_key)

    s.aggs.bucket(
        'scenes', 'nested', path='scenes'
    ).bucket(
        'by_location', 'terms', field='scenes.location.keyword', size=1000
    ).bucket(
        'by_season', 'reverse_nested'
    ).bucket(
        'season', 'terms', field='season'
    )
    
    return s


def agg_episodes_by_speaker(show_key: str, season: str = None, location: str = None, other_speaker: str = None) -> Search:
    print(f'begin agg_episodes_by_speaker for show_key={show_key} season={season} location={location} other_speaker={other_speaker}')

    # TODO this is nearly identical to agg_scenes_by_speaker, refactor?

    s = Search(index='transcripts')
    s = s.extra(size=0)

    s = s.filter('term', show_key=show_key)
    if season:
        s = s.filter('term', season=season)

    if location:
        s.aggs.bucket(
            'scenes', 'nested', path='scenes'
        ).bucket(
            'location_match', 'filter', filter={"match": {"scenes.location": location}}
        ).bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=1000
        ).bucket(
            'for_episode', 'reverse_nested' # TODO differs from agg_scenes_by_speaker
        )
    elif other_speaker:  # TODO location and other_speaker aren't exclusive of each other, this is just a WIP
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'speaker_match', 'filter', filter={"match": {"scenes.scene_events.spoken_by.keyword": other_speaker}}
        ).bucket(
            'for_scene', 'reverse_nested', path='scenes'
        ).bucket(
            'scene_events_2', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=1000
        ).bucket(
            'for_episode', 'reverse_nested' # TODO differs from agg_scenes_by_speaker
        )
    else:
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=1000
        ).bucket(
            'for_episode', 'reverse_nested' # TODO differs from agg_scenes_by_speaker
        )
    
    return s


def agg_episodes_by_location(show_key: str, season: str = None) -> Search:
    print(f'begin agg_episodes_by_speaker for show_key={show_key} season={season}')

    s = Search(index='transcripts')
    s = s.extra(size=0)

    s = s.filter('term', show_key=show_key)
    if season:
        s = s.filter('term', season=season)

    s.aggs.bucket(
        'scenes', 'nested', path='scenes'
    ).bucket(
        'by_location', 'terms', field='scenes.location.keyword', size=1000
    ).bucket(
        'by_episode', 'reverse_nested'
    )
    
    return s


def agg_scenes(show_key: str, season: str = None, episode_key: str = None, location: str = None) -> Search:
    print(f'begin agg_scenes for show_key={show_key} season={season} episode_key={episode_key} location={location}')

    s = Search(index='transcripts')
    s = s.extra(size=0)

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    # TODO location

    s.aggs.bucket('scene_count', 'sum', field='scene_count')

    return s


def agg_scenes_by_location(show_key: str, season: str = None, episode_key: str = None, speaker: str = None) -> Search:
    print(f'begin agg_scenes_by_location for show_key={show_key} season={season} episode_key={episode_key} speaker={speaker}')

    s = Search(index='transcripts')
    s = s.extra(size=0)

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    if speaker:
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'speaker_match', 'filter', filter={"match": {"scenes.scene_events.spoken_by.keyword": speaker}}
        ).bucket(
            'scenes', 'reverse_nested', path='scenes'
        ).bucket(
            'by_location', 'terms', field='scenes.location.keyword', size=1000)
    else:
        s.aggs.bucket(
            'scenes', 'nested', path='scenes'
        ).bucket(
            'by_location', 'terms', field='scenes.location.keyword', size=1000)

    return s


def agg_scenes_by_speaker(show_key: str, season: str = None, episode_key: str = None, 
                                location: str = None, other_speaker: str = None) -> Search:
    print(f'begin agg_scenes_by_speaker for show_key={show_key} season={season} episode_key={episode_key} location={location} other_speaker={other_speaker}')

    s = Search(index='transcripts')
    s = s.extra(size=0)

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    if location:
        s.aggs.bucket(
            'scenes', 'nested', path='scenes'
        ).bucket(
            'location_match', 'filter', filter={"match": {"scenes.location": location}}
        ).bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=1000
        ).bucket(
            'for_scene', 'reverse_nested', path='scenes'
        )
    elif other_speaker:  # TODO location and other_speaker aren't exclusive of each other, this is just a WIP
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'speaker_match', 'filter', filter={"match": {"scenes.scene_events.spoken_by.keyword": other_speaker}}
        ).bucket(
            'for_scene', 'reverse_nested', path='scenes'
        ).bucket(
            'scene_events_2', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=1000
        ).bucket(
            'for_scene_2', 'reverse_nested', path='scenes'
        )
    else:
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=1000
        ).bucket(
            'for_scene', 'reverse_nested', path='scenes'
        )
    
    return s


def agg_scene_events_by_speaker(show_key: str, season: str = None, episode_key: str = None, dialog: str = None) -> Search:
    print(f'begin agg_scene_events_by_speaker for show_key={show_key} season={season} episode_key={episode_key} dialog={dialog}')

    s = Search(index='transcripts')
    s = s.extra(size=0)

    # if dialog:
    #     nested_q = Q('nested', path='scenes.scene_events', 
    #             query=Q('match', **{'scenes.scene_events.dialog': dialog}),
    #             inner_hits={'size': 1000})

    #     s = s.query(nested_q)

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    '''
    IMPORTANT NOTE on what the `dialog_match` filter bucket is NOT:
    "dialog_match": {
        "filter": {
            "nested": {
                "path": "scenes.scene_events",
                "query": {
                    "match": {
                        "scenes.scene_events.dialog": dialog
                    }
                }
            }
        }
    }
    Applying that filter led to off-by-one mappings between scene_event speakers and dialog. F'ing bananas.
    '''

    if dialog:
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'dialog_match', 'filter', filter={"match": {"scenes.scene_events.dialog": dialog}}
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=1000)
    else:
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=1000)

    return s


def agg_dialog_word_counts(show_key: str, season: str = None, episode_key: str = None, speaker: str = None) -> Search:
    print(f'begin agg_dialog_word_counts for show_key={show_key} season={season} episode_key={episode_key} speaker={speaker}')

    s = Search(index='transcripts')
    s = s.extra(size=0)

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    if speaker:
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'speaker_match', 'filter', filter={'match': {'scenes.scene_events.spoken_by.keyword': speaker}}
        ).bucket(
            'word_count', 'sum', field='scenes.scene_events.dialog.word_count')
    else:
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=1000
        ).bucket(
            'word_count', 'sum', field='scenes.scene_events.dialog.word_count')
        
    return s


def keywords_by_episode(show_key: str, episode_key: str) -> dict:
    print(f'begin keywords_by_episode for show_key={show_key} episode_key={episode_key}')

    response = es_conn.termvectors(index='transcripts', id=f'{show_key}_{episode_key}', term_statistics='true', field_statistics='true',
                                   fields=['flattened_text'], filter={"max_num_terms": 1000, "min_term_freq": 1, "min_doc_freq": 1})

    return response


def keywords_by_corpus(show_key: str, season: str = None) -> dict:
    print(f'begin keywords_by_corpus for show_key={show_key} season={season}')

    keys = esr.fetch_doc_ids(ShowKey(show_key), season=season)

    if not keys:
        return {}
    
    response = es_conn.mtermvectors(index='transcripts', ids=keys['doc_ids'], term_statistics='true', field_statistics='true', fields=['flattened_text'])
    
    return response


def more_like_this(show_key: str, episode_key: str) -> Search:
    print(f'begin search_more_like_this for show_key={show_key} episode_key={episode_key}')

    s = Search(index='transcripts')
    s = s.extra(size=30)

    s = s.query(MoreLikeThis(like=[{'_index': 'transcripts', '_id': f'{show_key}_{episode_key}'}], fields=['flattened_text'],
                             max_query_terms=75, minimum_should_match='75%', min_term_freq=1, stop_words=STOPWORDS))
    
    s = s.source(excludes=['flattened_text', 'scenes'] + VECTOR_FIELDS + RELATIONS_FIELDS)
    
    return s


async def populate_focal_speakers(show_key: str, episode_key: str = None):
    print(f'begin populate_focal_speakers for show_key={show_key} episode_key={episode_key}')

    if episode_key:
        episode_doc_ids = [f'{show_key}_{episode_key}']
    else:
        doc_ids = esr.fetch_doc_ids(ShowKey(show_key))
        episode_doc_ids = doc_ids['doc_ids']
    
    episodes_to_focal_speakers = {}
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        episode_speakers = await esr.agg_scene_events_by_speaker(ShowKey(show_key), episode_key=episode_key)
        episode_focal_speakers = list(episode_speakers['scene_events_by_speaker'].keys())
        focal_speaker_count = min(len(episode_focal_speakers), 4)
        focal_speakers = episode_focal_speakers[1:focal_speaker_count]
        episodes_to_focal_speakers[episode_key] = focal_speakers

        es_episode = EsEpisodeTranscript.get(id=doc_id)
        es_episode.focal_speakers = focal_speakers
        save_es_episode(es_episode)

    return episodes_to_focal_speakers


async def populate_focal_locations(show_key: str, episode_key: str = None):
    print(f'begin populate_focal_locations for show_key={show_key} episode_key={episode_key}')

    if episode_key:
        episode_doc_ids = [f'{show_key}_{episode_key}']
    else:
        doc_ids = esr.fetch_doc_ids(ShowKey(show_key))
        episode_doc_ids = doc_ids['doc_ids']
    
    episodes_to_focal_locations = {}
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        episode_locations = await esr.agg_scenes_by_location(ShowKey(show_key), episode_key=episode_key)
        episode_focal_locations = list(episode_locations['scenes_by_location'].keys())
        focal_location_count = min(len(episode_focal_locations), 4)
        focal_locations = episode_focal_locations[1:focal_location_count]
        episodes_to_focal_locations[episode_key] = focal_locations

        es_episode = EsEpisodeTranscript.get(id=doc_id)
        es_episode.focal_locations = focal_locations
        save_es_episode(es_episode)

    return episodes_to_focal_locations


async def populate_relations(show_key: str, model_vendor: str, model_version: str, episodes_to_relations: dict, limit: int = None) -> dict:
    print(f'begin populate_relations for show_key={show_key} model vendor:version={model_vendor}:{model_version} len(episodes_to_relations)={len(episodes_to_relations)} limit={limit}')

    for doc_id, similar_episodes in episodes_to_relations.items():
        # sim_eps = [f"{sim_ep['episode_key']}|{sim_ep['score']}" for sim_ep in similar_episodes['matches']]
        # sim_eps = [(sim_ep['episode_key'], sim_ep['score']) for sim_ep in similar_episodes['matches']]
        sim_eps = {sim_ep['episode_key']:sim_ep['score'] for sim_ep in similar_episodes['matches']}

        # truncate response to limit (a more efficient way would be to limit the preceding query)
        if limit and limit < len(sim_eps):
            sim_eps = sim_eps[:limit]
        episodes_to_relations[doc_id] = sim_eps

        # write result to an `X_relations` field
        relations_field = f'{model_vendor}_{model_version}_relations_dict'
        es_episode = EsEpisodeTranscript.get(id=doc_id)
        es_episode[relations_field] = sim_eps
        save_es_episode(es_episode)

    return episodes_to_relations


def vector_search(show_key: str, vector_field: str, vectorized_qt: list, index_name: str = None, season: str = None) -> Search:
    print(f'begin vector_search for show_key={show_key} vector_field={vector_field} index_name={index_name} season={season}')

    if not index_name:
        index_name = 'transcripts'

    # s = Search(index='transcripts')
    # s = s.extra(size=1000)

    # s = s.filter('term', show_key=show_key)
    # if episode_key:
    #     s = s.filter('term', episode_key=episode_key)
    # if season:
    #     s = s.filter('term', season=season)
        
    # TODO hard-mapped based on number of TNG episodes, this needs to be passed into function (as episode count, speaker count, etc)
    k = 176

    knn_query = {
        "field": vector_field,
        "query_vector": vectorized_qt,
        "k": k,
        "num_candidates": k
    }

    filter_query = {
        "bool": {
            "filter": [
                {
                    "term": {
                        "show_key": show_key
                    }
                }
            ]
        }
    }

    if index_name == 'speakers':
        source = ['show_key', 'speaker']
    else:
        source = ['show_key', 'episode_key', 'title', 'season', 'sequence_in_season', 'air_date', 'scene_count', 'indexed_ts', 'focal_speakers', 'focal_locations']
    
    response = es_conn.knn_search(index=index_name, knn=knn_query, filter=filter_query, source=source)

    # s = s.query(index="transcripts", knn=knn_query, source=source)
    # print(f's.to_dict()={s.to_dict()}')
    # return s

    return response


def topic_vector_search(topic_grouping: str, vector_field: str, vectorized_qt: list) -> Search:
    print(f'begin topic_vector_search for topic_grouping={topic_grouping} vector_field={vector_field}')

    knn_query = {
        "field": vector_field,
        "query_vector": vectorized_qt,
        "k": 50,
        "num_candidates": 50
    }

    filter_query = {
        "bool": {
            "filter": [
                {
                    "term": {
                        "topic_grouping": topic_grouping
                    }
                }
            ]
        }
    }

    source = ['topic_grouping', 'topic_key', 'parent_key', 'topic_name', 'parent_name']

    response = es_conn.knn_search(index="topics", knn=knn_query, filter=filter_query, source=source)

    return response


def fetch_episode_embedding(show_key: str, episode_key: str, vector_field: str) -> Search:
    print(f'begin fetch_episode_embedding for show_key={show_key} episode_key={episode_key} vector_field={vector_field}')

    s = Search(index='transcripts')
    s = s.extra(size=1)

    s = s.filter('term', show_key=show_key)
    s = s.filter('term', episode_key=episode_key)
    s = s.source(includes=[vector_field])

    return s


def fetch_series_embeddings(show_key: str, vector_field: str) -> Search:
    print(f'begin fetch_series_embeddings for show_key={show_key} vector_field={vector_field}')

    s = Search(index='transcripts')
    s = s.extra(size=1000)

    s = s.filter('term', show_key=show_key)
    s = s.source(includes=[vector_field])

    return s


def fetch_topic_embedding(topic_grouping: str, topic_key: str, vector_field: str) -> Search:
    print(f'begin fetch_topic_embedding for topic_grouping={topic_grouping} topic_key={topic_key} vector_field={vector_field}')

    s = Search(index='topics')
    s = s.extra(size=1)

    s = s.filter('term', topic_grouping=topic_grouping)
    s = s.filter('term', topic_key=topic_key)
    s = s.source(includes=[vector_field])

    return s
