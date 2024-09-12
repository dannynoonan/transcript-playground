# from datetime import datetime
# from elasticsearch import Elasticsearch
# from elasticsearch import RequestsHttpConnection
from elasticsearch_dsl import Search, connections, Q, A
from elasticsearch_dsl.query import MoreLikeThis

from app.config import settings
from app.es.es_metadata import STOPWORDS, VECTOR_FIELDS, RELATIONS_FIELDS
from app.es.es_model import (
    EsEpisodeTranscript, EsEpisodeNarrativeSequence, EsSpeaker, EsSpeakerSeason, EsSpeakerEpisode, 
    EsSpeakerUnified, EsTopic, EsEpisodeTopic, EsSpeakerTopic, EsSpeakerSeasonTopic, EsSpeakerEpisodeTopic
)
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
    es_conn.indices.put_settings(index="transcripts", body={"index.mapping.total_fields.limit": 10000})


def init_narratives_index():
    EsEpisodeNarrativeSequence.init()
    es_conn.indices.put_settings(index="narratives", body={"index": {"max_inner_result_window": 1000}})


def init_speakers_index():
    EsSpeaker.init()
    es_conn.indices.put_settings(index="speakers", body={"index": {"max_inner_result_window": 1000}})


def init_speaker_seasons_index():
    EsSpeakerSeason.init()
    es_conn.indices.put_settings(index="speaker_seasons", body={"index": {"max_inner_result_window": 1000}})


def init_speaker_episodes_index():
    EsSpeakerEpisode.init()
    es_conn.indices.put_settings(index="speaker_episodes", body={"index": {"max_inner_result_window": 1000}})


def init_speaker_unified_index():
    EsSpeakerUnified.init()
    es_conn.indices.put_settings(index="speaker_embeddings_unified", body={"index": {"max_inner_result_window": 1000}})


def init_topics_index():
    EsTopic.init()
    es_conn.indices.put_settings(index="topics", body={"index": {"max_inner_result_window": 1000}})


def init_episode_topics_index():
    EsEpisodeTopic.init()
    es_conn.indices.put_settings(index="episode_topics", body={"index": {"max_inner_result_window": 1000}})


def init_speaker_topics_index():
    EsSpeakerTopic.init()
    es_conn.indices.put_settings(index="speaker_topics", body={"index": {"max_inner_result_window": 1000}})


def init_speaker_season_topics_index():
    EsSpeakerSeasonTopic.init()
    es_conn.indices.put_settings(index="speaker_season_topics", body={"index": {"max_inner_result_window": 1000}})


def init_speaker_episode_topics_index():
    EsSpeakerEpisodeTopic.init()
    es_conn.indices.put_settings(index="speaker_episode_topics", body={"index": {"max_inner_result_window": 1000}})


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


def save_episode_narrative(es_episode_narrative: EsEpisodeNarrativeSequence) -> None:
    # TODO this is functionally identical to `save_es_episode`, do we even need either of them?
    es_episode_narrative.save()


def save_es_speaker(es_speaker: EsSpeaker) -> None:
    es_speaker.save()
    es_speaker_unified = EsSpeakerUnified(show_key=es_speaker.show_key, speaker=es_speaker.speaker, 
                                          layer_key='SERIES', word_count=es_speaker.word_count,
                                          openai_ada002_embeddings=es_speaker.openai_ada002_embeddings)
    es_speaker_unified.save()


def save_es_speaker_season(es_speaker_season: EsSpeakerSeason) -> None:
    es_speaker_season.save()
    es_speaker_unified = EsSpeakerUnified(show_key=es_speaker_season.show_key, speaker=es_speaker_season.speaker, 
                                          layer_key=f'S{es_speaker_season.season}', word_count=es_speaker_season.word_count,
                                          openai_ada002_embeddings=es_speaker_season.openai_ada002_embeddings)
    es_speaker_unified.save()


def save_es_speaker_episode(es_speaker_episode: EsSpeakerEpisode) -> None:
    es_speaker_episode.save()
    es_speaker_unified = EsSpeakerUnified(show_key=es_speaker_episode.show_key, speaker=es_speaker_episode.speaker, 
                                          layer_key=f'S{es_speaker_episode.season}E{es_speaker_episode.sequence_in_season}', 
                                          word_count=es_speaker_episode.word_count,
                                          openai_ada002_embeddings=es_speaker_episode.openai_ada002_embeddings)
    es_speaker_unified.save()


def fetch_episode_by_key(show_key: str, episode_key: str, all_fields: bool = False) -> Search:
    # print(f'begin fetch_episode_by_key for show_key={show_key} episode_key={episode_key}')

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


def fetch_episode_narrative(show_key: str, episode_key: str, speaker_group: str) -> EsEpisodeNarrativeSequence|None:
    print(f'begin fetch_episode_narrative for show_key={show_key} episode_key={episode_key} speaker_group={speaker_group}')

    doc_id = f'{show_key}_{episode_key}_{speaker_group}'

    try:
        ep_narr = EsEpisodeNarrativeSequence.get(id=doc_id, index='narratives')
    except Exception as e:
        print(f'Failed to fetch episode narrative for show_key=`{show_key}` episode_key=`{episode_key}` speaker_group=`{speaker_group}`')
        return None

    return ep_narr


def fetch_narrative_sequences(show_key: str, episode_key: str) -> Search:
    # print(f'begin fetch_narrative_sequences for show_key={show_key} season={episode_key}')

    s = Search(index='narratives')
    s = s.extra(size=1000)

    s = s.filter('term', show_key=show_key)
    s = s.filter('term', episode_key=episode_key)

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
    # if speaker.child_topics:
    #     speaker.child_topics = speaker.child_topics._d_
    # if speaker.parent_topics:
    #     speaker.parent_topics = speaker.parent_topics._d_

    return speaker


def fetch_indexed_speakers(show_key: str, season: int = None, return_fields: list = None, min_episode_count: int = None) -> Search:
    print(f'begin fetch_indexed_speakers for show_key={show_key}')

    s = Search(index='speakers')
    s = s.extra(size=1000)

    s = s.filter('term', show_key=show_key)
    # NOTE this filter is misleading: it will exclude speakers by season but will not adjust aggregate series-level counts
    if season:
        s = s.filter('term', season=str(season))

    if min_episode_count:
        s = s.filter('range', episode_count={'gte': min_episode_count})

    s = s.sort({"episode_count" : {"order" : "desc"}}, {"scene_count" : {"order" : "desc"}})

    if return_fields:
        s = s.source(includes=return_fields)

    return s


def search_speakers(qt: str, show_key: str, return_fields: list = None) -> Search:
    print(f'begin search_speaker for qt={qt} show_key={show_key}')

    s = Search(index='speakers')
    s = s.extra(size=100)

    # TODO this gets around limits on searching keyword `speaker` field, but true solution would be to add this to `alt_names` or `searchable_names`
    qt_upper = qt.upper()

    q = Q('bool', minimum_should_match=1, should=[
            Q('match', speaker=qt_upper), Q('match', alt_names=qt), Q('match', actor_names=qt)])

    s = s.query(q)

    if show_key:
        s = s.filter('term', show_key=show_key)

    if return_fields:
        s = s.source(includes=return_fields)

    return s


def fetch_speaker_season(show_key: str, speaker_name: str, season: int) -> EsSpeakerSeason|None:
    print(f'begin fetch_speaker_season for show_key={show_key} speaker_name={speaker_name} season={season}')

    doc_id = f'{show_key}_{speaker_name}_{season}'

    try:
        speaker_season = EsSpeakerSeason.get(id=doc_id, index='speaker_seasons')
    except Exception as e:
        print(f'Failed to fetch speaker_season for show_key={show_key} speaker={speaker_name} season={season}', e)
        return None

    return speaker_season


def fetch_speaker_seasons(show_key: str, speaker: str = None, season: int = None, return_fields: list = []) -> Search:
    print(f'begin fetch_speaker_seasons for show_key={show_key} speaker={speaker} season={season} return_fields={return_fields}')

    if not (speaker or season):
        raise Exception('fetch_speaker_seasons requires at least `speaker` or `season')

    s = Search(index='speaker_seasons')
    s = s.extra(size=1000)

    s = s.filter('term', show_key=show_key)
    if speaker:
        s = s.filter('term', speaker=speaker)
    if season:
        s = s.filter('term', season=season)

    s = s.sort('season')

    if return_fields:
        s = s.source(includes=return_fields)

    return s


def fetch_speaker_episode(show_key: str, speaker_name: str, episode_key: str) -> EsSpeakerEpisode|None:
    print(f'begin fetch_speaker_episode for show_key={show_key} speaker_name={speaker_name} episode_key={episode_key}')

    doc_id = f'{show_key}_{speaker_name}_{episode_key}'

    try:
        speaker_episode = EsSpeakerEpisode.get(id=doc_id, index='speaker_episodes')
    except Exception as e:
        print(f'Failed to fetch speaker_episode for show_key={show_key} speaker={speaker_name} episode_key={episode_key}', e)
        return None

    return speaker_episode


def fetch_speaker_episodes(show_key: str, speaker: str = None, episode_key: str = None, season: int = None, return_fields: list = []) -> Search:
    print(f'begin fetch_speaker_episodes for show_key={show_key} episode_key={episode_key} speaker={speaker} season={season} return_fields={return_fields}')

    if not (speaker or episode_key):
        raise Exception('fetch_speaker_episodes requires at least `speaker` or `episode_key')

    s = Search(index='speaker_episodes')
    s = s.extra(size=1000)

    s = s.filter('term', show_key=show_key)
    if speaker:
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


def fetch_speaker_embeddings(show_key: str, speaker: str, vector_field: str, min_depth: bool = True) -> tuple[list, dict, dict]:
    print(f'begin fetch_speaker_embeddings for show_key={show_key} speaker={speaker} vector_field={vector_field}')

    try:
        es_speaker = fetch_speaker(show_key, speaker)
        speaker_series_embeddings = getattr(es_speaker, vector_field)
    except Exception as e:
        return {"error": f"Failed fetch_speaker for show_key={show_key} speaker={speaker}: {e}"}
    
    # min_depth: if we found series-level embeddings, return them - we're done 
    if min_depth and speaker_series_embeddings:
         return speaker_series_embeddings, {}, {}
            
    seasons = es_speaker.seasons_to_episode_keys._d_.keys()

    all_speaker_season_embeddings = {}
    for season in seasons:
        try:
            es_speaker_season = fetch_speaker_season(show_key, speaker, season)
            if not es_speaker_season:
                print(f"Failed fetch_speaker_episode for show_key={show_key} speaker={speaker} season={season}")
                continue
            speaker_season_embeddings = getattr(es_speaker_season, vector_field)
            if speaker_season_embeddings:
                all_speaker_season_embeddings[season] = speaker_season_embeddings
        except Exception as e:
            print(f"Failed fetch_speaker_season for show_key={show_key} speaker={speaker} season={season}: {e}")

    # min_depth: if we found season-level embeddings for all seasons, return them - we're done 
    if min_depth and len(all_speaker_season_embeddings) == len(seasons):
        return speaker_series_embeddings, all_speaker_season_embeddings, {}
    
    all_speaker_episode_embeddings = {}
    for season, episode_keys in es_speaker.seasons_to_episode_keys._d_.items():
        # min_depth: skip episodes in seasons we've already fetched embeddings for
        if min_depth and season in all_speaker_season_embeddings:
            continue
        for episode_key in episode_keys:
            try:
                es_speaker_episode = fetch_speaker_episode(show_key, speaker, episode_key)
                if not es_speaker_episode:
                    print(f"Failed fetch_speaker_episode for show_key={show_key} speaker={speaker} episode_key={episode_key}")
                    continue
                speaker_episode_embeddings = getattr(es_speaker_episode, vector_field)
                if speaker_episode_embeddings:
                    all_speaker_episode_embeddings[episode_key] = speaker_episode_embeddings
            except Exception as e:
                print(f"Failed fetch_speaker_episode for show_key={show_key} speaker={speaker} episode_key={episode_key}: {e}")

    return speaker_series_embeddings, all_speaker_season_embeddings, all_speaker_episode_embeddings


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
    print(f'begin fetch_topic_grouping for topic_grouping={topic_grouping}')

    s = Search(index='topics')
    s = s.extra(size=1000)

    s = s.filter('term', topic_grouping=topic_grouping)

    s = s.sort('parent_key', 'topic_key')

    if return_fields:
        s = s.source(includes=return_fields)

    return s


def fetch_episode_topic(show_key: str, episode_key: str, topic_grouping: str, topic_key: str, model_vendor: str, model_version: str) -> EsEpisodeTopic|None:
    doc_id = f'{show_key}_{episode_key}_{topic_grouping}_{topic_key}_{model_vendor}_{model_version}'
    print(f'begin fetch_episode_topic for doc_id={doc_id}')

    try:
        episode_topic = EsEpisodeTopic.get(id=doc_id, index='episode_topics')
    except Exception as e:
        print(f'Failed to fetch episode_topic with doc_id={doc_id}')
        return None

    return episode_topic


def fetch_episode_topics(show_key: str, episode_key: str, topic_grouping: str, level: str = None, limit: int = None, sort_by: str = None) -> Search:
    print(f'begin fetch_speaker_episode_topics for show_key={show_key} episode_key={episode_key} topic_grouping={topic_grouping} level={level}')

    if not limit:
        limit = 100
    if not sort_by:
        sort_by = 'score'

    s = Search(index='episode_topics')
    s = s.extra(size=limit)

    s = s.filter('match', show_key=show_key)
    s = s.filter('term', episode_key=episode_key)
    s = s.filter('match', topic_grouping=topic_grouping)
    if level:
        if level in ['parent', 'root', 'top']:
            s = s.filter('term', is_parent=True)
        elif level in ['child', 'leaf']:
            s = s.filter('term', is_parent=False)

    s = s.sort({sort_by: {'order': 'desc'}})

    return s


def fetch_speaker_topics(speaker: str, show_key: str, topic_grouping: str, level: str = None, limit: int = None) -> Search:
    print(f'begin fetch_speaker_topics for speaker={speaker} show_key={show_key} topic_grouping={topic_grouping} level={level}')

    if not limit:
        limit = 100

    s = Search(index='speaker_topics')
    s = s.extra(size=limit)

    s = s.filter('term', speaker=speaker)
    s = s.filter('term', show_key=show_key)
    s = s.filter('term', topic_grouping=topic_grouping)
    if level:
        if level in ['parent', 'root', 'top']:
            s = s.filter('term', is_parent=True)
        elif level in ['child', 'leaf']:
            s = s.filter('term', is_parent=False)

    s = s.sort({'score': {'order': 'desc'}})

    return s


def fetch_speaker_season_topics(show_key: str, topic_grouping: str, speaker: str = None, season: int = None, level: str = None, limit: int = None) -> Search:
    print(f'begin fetch_speaker_season_topics for show_key={show_key} topic_grouping={topic_grouping} speaker={speaker} season={season} level={level}')

    if not (speaker or season):
        raise Exception(f'Failure to fetch_speaker_season_topics: both `speaker` and `season` were None, at least one must be set')
    
    if not limit:
        limit = 1000

    s = Search(index='speaker_season_topics')
    s = s.extra(size=1000)

    s = s.filter('term', show_key=show_key)
    s = s.filter('term', topic_grouping=topic_grouping)
    if speaker:
        s = s.filter('term', speaker=speaker)
    if season:
        s = s.filter('term', season=season)
    if level:
        if level in ['parent', 'root', 'top']:
            s = s.filter('term', is_parent=True)
        elif level in ['child', 'leaf']:
            s = s.filter('term', is_parent=False)

    s = s.sort({'season': {'order': 'asc'}}, {'score': {'order': 'desc'}})

    return s


def fetch_speaker_episode_topics(speaker: str, show_key: str, topic_grouping: str, episode_key: str = None, season: int = None, 
                                 level: str = None, limit: int = None) -> Search:
    print(f'begin fetch_speaker_episode_topics for speaker={speaker} show_key={show_key} topic_grouping={topic_grouping} episode_key={episode_key} season={season} level={level}')

    if not limit:
        limit = 1000

    s = Search(index='speaker_episode_topics')
    # s = s.extra(size=limit)
    s = s.extra(size=10000)

    s = s.filter('term', speaker=speaker)
    s = s.filter('term', show_key=show_key)
    s = s.filter('term', topic_grouping=topic_grouping)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)
    if level:
        if level in ['parent', 'root', 'top']:
            s = s.filter('term', is_parent=True)
        elif level in ['child', 'leaf']:
            s = s.filter('term', is_parent=False)

    s = s.sort({'season': {'order': 'asc'}}, {'episode_key': {'order': 'asc'}}, {'score': {'order': 'desc'}})

    return s


def search_episode_topics(show_key: str, topic_grouping: str, topic_key: str, season: int = None, limit: int = None, sort_by: str = None) -> Search:
    """
    Find episodes by topic via episode_topics index
    """
    print(f'begin search_episode_topics for show_key={show_key} topic={topic_grouping}:{topic_key} season={season}')

    if not limit:
        limit = 100
    if not sort_by:
        sort_by = 'score'

    s = Search(index='episode_topics')
    s = s.extra(size=limit)

    s = s.filter('match', show_key=show_key)
    s = s.filter('match', topic_grouping=topic_grouping)
    s = s.filter('match', topic_key=topic_key)
    if season:
        s = s.filter('match', season=season)

    s = s.sort({sort_by: {'order': 'desc'}})

    return s


def search_speaker_topics(topic_grouping: str, topic_key: str, show_key: str = None, limit: int = None, min_word_count: int = None) -> Search:
    """
    Find speakers by topic via speaker_topics index
    """
    print(f'begin search_speaker_topics for topic={topic_grouping}:{topic_key} show_key={show_key}')

    if not limit:
        limit = 100

    s = Search(index='speaker_topics')
    s = s.extra(size=limit)

    s = s.filter('match', topic_grouping=topic_grouping)
    s = s.filter('match', topic_key=topic_key)
    if show_key:
        s = s.filter('match', show_key=show_key)
    if min_word_count:
        s = s.filter('range', word_count={'gt': min_word_count})

    s = s.sort({'score': {'order': 'desc'}})

    return s


def search_speaker_season_topics(topic_grouping: str, topic_key: str, show_key: str, season: int = None, limit: int = None, min_word_count: int = None) -> Search:
    """
    Find speaker_seasons by topic via speaker_season_topics index
    """
    print(f'begin search_speaker_season_topics for topic={topic_grouping}:{topic_key} show_key={show_key} season={season}')

    if not limit:
        limit = 100

    s = Search(index='speaker_season_topics')
    s = s.extra(size=limit)

    s = s.filter('match', topic_grouping=topic_grouping)
    s = s.filter('match', topic_key=topic_key)
    s = s.filter('match', show_key=show_key)
    if season:
        s = s.filter('match', season=season)
    if min_word_count:
        s = s.filter('range', word_count={'gt': min_word_count})

    s = s.sort({'score': {'order': 'desc'}})

    return s


def search_speaker_episode_topics(topic_grouping: str, topic_key: str, show_key: str, season: int = None, episode_key: str = None, limit: int = None, 
                                  min_word_count: int = None) -> Search:
    """
    Find speaker_episodes by topic via speaker_episode_topics index
    """
    print(f'begin search_speaker_episode_topics for topic={topic_grouping}:{topic_key} show_key={show_key} season={season} episode_key={episode_key}')

    if not limit:
        limit = 100

    s = Search(index='speaker_episode_topics')
    s = s.extra(size=limit)

    s = s.filter('match', topic_grouping=topic_grouping)
    s = s.filter('match', topic_key=topic_key)
    s = s.filter('match', show_key=show_key)
    if season:
        s = s.filter('match', season=season)
    if episode_key:
        s = s.filter('match', episode_key=episode_key)
    if min_word_count:
        s = s.filter('range', word_count={'gt': min_word_count})

    s = s.sort({'score': {'order': 'desc'}})

    return s


@DeprecationWarning
def search_speakers_by_topic(topic_grouping: str, topic_key: str, is_parent: bool = False, show_key: str = None, min_word_count: int = None) -> Search:
    print(f'begin search_speakers_by_topic for topic={topic_grouping}:{topic_key} is_parent={is_parent} show_key={show_key} min_word_count={min_word_count}')

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
    # print(f'begin search_scene_events_multi_speaker for show_key={show_key} season={season} episode_key={episode_key} speakers={speakers}')

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


def populate_focal_speakers(show_key: str, episode_key: str = None):
    print(f'begin populate_focal_speakers for show_key={show_key} episode_key={episode_key}')

    if episode_key:
        episode_doc_ids = [f'{show_key}_{episode_key}']
    else:
        doc_ids = esr.fetch_doc_ids(ShowKey(show_key))
        episode_doc_ids = doc_ids['doc_ids']
    
    episodes_to_focal_speakers = {}
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        episode_speakers = esr.agg_scene_events_by_speaker(ShowKey(show_key), episode_key=episode_key)
        episode_focal_speakers = list(episode_speakers['scene_events_by_speaker'].keys())
        focal_speaker_count = min(len(episode_focal_speakers), 4)
        focal_speakers = episode_focal_speakers[1:focal_speaker_count]
        episodes_to_focal_speakers[episode_key] = focal_speakers

        es_episode = EsEpisodeTranscript.get(id=doc_id)
        es_episode.focal_speakers = focal_speakers
        save_es_episode(es_episode)

    return episodes_to_focal_speakers


def populate_focal_locations(show_key: str, episode_key: str = None):
    print(f'begin populate_focal_locations for show_key={show_key} episode_key={episode_key}')

    if episode_key:
        episode_doc_ids = [f'{show_key}_{episode_key}']
    else:
        doc_ids = esr.fetch_doc_ids(ShowKey(show_key))
        episode_doc_ids = doc_ids['doc_ids']
    
    episodes_to_focal_locations = {}
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        episode_locations = esr.agg_scenes_by_location(ShowKey(show_key), episode_key=episode_key)
        episode_focal_locations = list(episode_locations['scenes_by_location'].keys())
        focal_location_count = min(len(episode_focal_locations), 4)
        focal_locations = episode_focal_locations[1:focal_location_count]
        episodes_to_focal_locations[episode_key] = focal_locations

        es_episode = EsEpisodeTranscript.get(id=doc_id)
        es_episode.focal_locations = focal_locations
        save_es_episode(es_episode)

    return episodes_to_focal_locations


def populate_episode_topics(show_key: str, episode: EsEpisodeTranscript, topics: list, model_vendor: str, model_version: str) -> list:
    print(f'begin populate_episode_topics for show_key={show_key} episode_key={episode.episode_key} len(topics)={len(topics)}')
    if not topics:
        return []
    es_episode_topics = []
    # topics arrive sorted desc by score
    high_score = topics[0]['score']
    low_score = topics[len(topics)-1]['score']
    score_range = high_score - low_score
    for topic in topics:
        if score_range > 0:
            topic['dist_score'] = (topic['score'] - low_score) / score_range
        else:
            topic['dist_score'] = None
        is_parent = 'parent_key' not in topic or topic['parent_key'] == ''
        es_episode_topic = EsEpisodeTopic(show_key=show_key, episode_key=episode.episode_key, episode_title=episode.title, season=episode.season, 
                                          sequence_in_season=episode.sequence_in_season, air_date=episode.air_date, topic_grouping=topic['topic_grouping'], 
                                          topic_key=topic['topic_key'], topic_name=topic['topic_name'], raw_score=topic['score'], score=topic['dist_score'], 
                                          is_parent=is_parent, model_vendor=model_vendor, model_version=model_version)
        es_episode_topics.append(es_episode_topic)
        es_episode_topic.save()

    return es_episode_topics


def populate_speaker_topics(show_key: str, speaker: str, es_speaker: EsSpeaker, topics: list, model_vendor: str, model_version: str) -> list:
    print(f'begin populate_speaker_topics for show_key={show_key} speaker={speaker} len(topics)={len(topics)}')
    if not topics:
        return []
    es_speaker_topics = []
    # topics arrive sorted desc by score
    if 'dist_score' not in topics[0]:
        high_score = topics[0]['score']
        low_score = topics[len(topics)-1]['score']
        score_range = high_score - low_score
    for topic in topics:
        # TODO this got hackey due to score normalization being added late
        # this assumes that if first topic had dist_score then all of them did
        if 'dist_score' not in topic:
            topic['dist_score'] = (topic['score'] - low_score) / score_range
        if 'score' not in topic:
            topic['score'] = -1
        is_parent = 'parent_key' not in topic or topic['parent_key'] == ''
        is_aggregate = 'is_aggregate' in topic and topic['is_aggregate']
        es_speaker_topic = EsSpeakerTopic(show_key=show_key, speaker=speaker, word_count=es_speaker.word_count, topic_grouping=topic['topic_grouping'], 
                                          topic_key=topic['topic_key'], topic_name=topic['topic_name'], raw_score=topic['score'], score=topic['dist_score'],
                                          is_parent=is_parent, is_aggregate=is_aggregate, model_vendor=model_vendor, model_version=model_version)
        es_speaker_topics.append(es_speaker_topic)
        es_speaker_topic.save()

    return es_speaker_topics


def populate_speaker_season_topics(show_key: str, speaker: str, speaker_season: EsSpeakerSeason, topics: list, model_vendor: str, model_version: str) -> list:
    print(f'begin populate_speaker_season_topics for show_key={show_key} speaker={speaker} season={speaker_season.season} len(topics)={len(topics)}')
    if not topics:
        return []
    es_speaker_season_topics = []
    # topics arrive sorted desc by score
    if 'dist_score' not in topics[0]:
        high_score = topics[0]['score']
        low_score = topics[len(topics)-1]['score']
        score_range = high_score - low_score
    for topic in topics:
        # TODO this got hackey due to score normalization being added late
        # this assumes that if first topic had dist_score then all of them did
        if 'dist_score' not in topic:
            topic['dist_score'] = (topic['score'] - low_score) / score_range
        if 'score' not in topic:
            topic['score'] = -1
        is_parent = 'parent_key' not in topic or topic['parent_key'] == ''
        is_aggregate = 'is_aggregate' in topic and topic['is_aggregate']
        es_speaker_season_topic = EsSpeakerSeasonTopic(show_key=show_key, speaker=speaker, season=speaker_season.season, word_count=speaker_season.word_count, 
                                                       topic_grouping=topic['topic_grouping'], topic_key=topic['topic_key'], topic_name=topic['topic_name'], 
                                                       raw_score=topic['score'], score=topic['dist_score'], is_parent=is_parent, is_aggregate=is_aggregate,
                                                       model_vendor=model_vendor, model_version=model_version)
        es_speaker_season_topics.append(es_speaker_season_topic)
        es_speaker_season_topic.save()

    return es_speaker_season_topics


def populate_speaker_episode_topics(show_key: str, speaker: str, speaker_episode: EsSpeakerEpisode, topics: list, model_vendor: str, model_version: str) -> list:
    print(f'begin populate_speaker_season_topics for show_key={show_key} speaker={speaker} episode_key={speaker_episode.episode_key} len(topics)={len(topics)}')
    if not topics:
        return []
    es_speaker_episode_topics = []
    # topics arrive sorted desc by score
    high_score = topics[0]['score']
    low_score = topics[len(topics)-1]['score']
    score_range = high_score - low_score
    for topic in topics:
        topic['dist_score'] = (topic['score'] - low_score) / score_range
        is_parent = 'parent_key' not in topic or topic['parent_key'] == ''
        es_speaker_episode_topic = EsSpeakerEpisodeTopic(show_key=show_key, speaker=speaker, episode_key=speaker_episode.episode_key, 
                                                         episode_title=speaker_episode.title, season=speaker_episode.season, 
                                                         sequence_in_season=speaker_episode.sequence_in_season, air_date=speaker_episode.air_date, 
                                                         word_count=speaker_episode.word_count, topic_grouping=topic['topic_grouping'], 
                                                         topic_key=topic['topic_key'], topic_name=topic['topic_name'], raw_score=topic['score'],
                                                         score=topic['dist_score'], is_parent=is_parent, model_vendor=model_vendor, model_version=model_version)
        es_speaker_episode_topics.append(es_speaker_episode_topic)
        es_speaker_episode_topic.save()

    return es_speaker_episode_topics


def populate_relations(show_key: str, model_vendor: str, model_version: str, episodes_to_relations: dict, limit: int = None) -> dict:
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


def vector_search(show_key: str, vector_field: str, vectorized_qt: list, index_name: str = None, min_word_count: int = None, season: str = None) -> Search:
    print(f'begin vector_search for show_key={show_key} vector_field={vector_field} index_name={index_name} min_word_count={min_word_count} season={season}')

    if not index_name:
        index_name = 'transcripts'

    # TODO hard-mapped based on number of TNG episodes / arbitary speaker count, need to calculate this or pass as parameter
    if index_name == 'transcripts':
        k = 176
    elif index_name in ['speakers', 'speaker_seasons', 'speaker_episodes']:
        k = 50
    elif index_name == 'speaker_embeddings_unified':
        k = 100
    else:
        k = 176

    # s = Search(index='transcripts')
    # s = s.extra(size=1000)

    # s = s.filter('term', show_key=show_key)
    # if episode_key:
    #     s = s.filter('term', episode_key=episode_key)
    # if season:
    #     s = s.filter('term', season=season)
    
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

    if min_word_count:
        min_wc_filer = dict(range=dict(word_count=dict(gte=min_word_count)))
        filter_query['bool']['filter'].append(min_wc_filer)

    if index_name == 'speakers':
        source = ['show_key', 'speaker']
    elif index_name == 'speaker_seasons':
        source = ['show_key', 'season', 'speaker']
    elif index_name == 'speaker_episodes':
        source = ['show_key', 'episode_key', 'season', 'sequence_in_season', 'speaker']
    elif index_name == 'speaker_embeddings_unified':
        source = ['show_key', 'layer_key', 'speaker']
    else:
        source = ['show_key', 'episode_key', 'title', 'season', 'sequence_in_season', 'air_date', 'scene_count', 'indexed_ts', 'focal_speakers', 'focal_locations', 
                  'topics_universal', 'topics_focused', 'topics_universal_tfidf', 'topics_focused_tfidf']
    
    print(f'filter_query={filter_query}')
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


def save_episode_sentiment(es_episode: EsEpisodeTranscript, episode_sentiment: dict) -> None:
    print(f'begin save_episode_sentiment for es_episode={es_episode.title}')

    es_episode.nltk_sent_pos = episode_sentiment['pos']
    es_episode.nltk_sent_neg = episode_sentiment['neg']
    es_episode.nltk_sent_neu = episode_sentiment['neu']

    es_episode.save()

