# from datetime import datetime
from elasticsearch import Elasticsearch
# from elasticsearch import RequestsHttpConnection
# from elasticsearch_dsl import Search, connections
from elasticsearch_dsl import Search, Q
from elasticsearch_dsl.query import MultiMatch

from config import settings
from es_model import EsEpisodeTranscript, EsScene, EsSceneEvent


es_client = Elasticsearch(
    hosts=[{'host': settings.es_host, 'port': settings.es_port, 'scheme': 'https'}],    
    basic_auth=(settings.es_user, settings.es_pass),
    verify_certs=False
    # connection_class=RequestsHttpConnection
)

# s = Search(using=es_client, index='transcripts')

# connections.create_connection(hosts=['http://localhost:9200'], timeout=20)

# connections.configure(
#     default={'hosts': 'http://localhost:9200'},
    # dev={
    #     'hosts': ['http://localhost:9200'],
    #     'sniff_on_start': True
    # }
# )


def init_mappings():
    EsSceneEvent.init()
    EsScene.init()
    EsEpisodeTranscript.init()


async def save_es_episode(es_episode: EsEpisodeTranscript) -> None:
    es_episode.save(using=es_client)
    # persisted_es_episode = EsEpisodeTranscript.get(id=es_episode.meta.id, ignore=404)
    # if persisted_es_episode:
    #     es_episode.update(using=es, doc_as_upsert=True)
    # else:
    #     es_episode.save(using=es)


async def search_episodes_by_qt(show_key: str, qt: str, fields: list = None) -> list:
    if not fields:
        fields = ['title', 'episode_key']

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

    # MultiMatch(query=query_term, fields=['title', 'episode_key'])

    # q = Q("multi_match", query=query_term, fields=['title', 'episode_key'])
    q = Q('bool', must=[Q('match', show_key=show_key), Q('multi_match', query=qt, fields=fields)])
    # s.query(q)
    # s.extra(track_total_hits=True, size=0)
    # s = s.extra(track_total_hits=True)
    s = s.source(excludes=['scenes'])
    s = s.query(q)
    # s = s.query("multi_match", query=query_term, fields=['title', 'episode_key'])
    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    s = s.execute()

    print('*************************************************')
    print(f'response.to_dict()={s.to_dict()}')
    print('*************************************************')

    for hit in s.hits:
        results.append(hit)
    return results


async def search_scenes_by_qt(show_key: str, qt: str, fields: list = None) -> list:
    if not fields:
        fields = ['scenes.location', 'scenes.description']

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

    q = Q('bool', must=[Q('match', show_key=show_key), Q('multi_match', query=qt, fields=fields)])
    s = s.source(excludes=['scenes.scene_events'])
    s = s.query(q)

    s = s.execute()

    for hit in s.hits:
        results.append(hit)
    return results


async def search_scenes_by_location(show_key: str, location: str) -> list:
    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

    # q = Q('nested', path='scenes', query=Q('match', **{'scenes.location': location}))
    # q = Q('bool', must=[Q('match', show_key=show_key), Q('multi_match', query=qt, fields=fields)])
    # s = s.query('nested', path='scenes', query=Q('match', **{'scenes.location': location}))
    # s = s.query('nested', path='scenes', query=Q('match', scenes__location={'term': location}))
    # s = s.query('nested', path='scenes', query=Q('term', scenes__location=location))

    q = Q("match", scenes__location=location)
    s = s.source(excludes=['scenes.scene_events'])
    s = s.query(q)

    s = s.execute()

    for hit in s.hits:
        results.append(hit)
    return results


async def search_scene_events_by_qt(show_key: str, qt: str, fields: list = None) -> list:
    if not fields:
        fields = ['scenes.scene_events.dialogue_spoken_by', 'scenes.scene_events.context_info']

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

    q = Q('bool', must=[Q('match', show_key=show_key), Q('multi_match', query=qt, fields=fields)])
    s = s.source(excludes=['scenes'])
    s = s.query(q)

    s = s.execute()

    for hit in s.hits:
        results.append(hit)
    return results


async def agg_episodes_by_location(show_key: str) -> list:
    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = {}

    q = Q('bool', must=[Q('match', show_key=show_key)])
    s = s.query(q)
    s.aggs.bucket(f'by_location', 'terms', field='scenes.location.keyword', size=1000)

    s = s.execute()

    # print('*************************************************')
    # print(f's.aggregations.by_location={s.aggregations.by_location}')
    # print('*************************************************')
    # print(f's.hits.total={s.hits.total}')
    # print('*************************************************')
    # print(f's.aggregations.by_location.buckets={s.aggregations.by_location.buckets}')
    # print('*************************************************')

    for item in s.aggregations.by_location.buckets:
        results[item.key] = item.doc_count

    return results


async def agg_episodes_by_character(show_key: str) -> list:
    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = {}

    q = Q('bool', must=[Q('match', show_key=show_key)])
    s = s.query(q)
    s.aggs.bucket(f'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=1000)

    s = s.execute()

    # print('*************************************************')
    # print(f's.aggregations.by_speaker={s.aggregations.by_speaker}')
    # print('*************************************************')
    # print(f's.hits.total={s.hits.total}')
    # print('*************************************************')
    # print(f's.aggregations.by_speaker.buckets={s.aggregations.by_speaker.buckets}')
    # print('*************************************************')

    for item in s.aggregations.by_speaker.buckets:
        results[item.key] = item.doc_count

    return results


# doc = {
#     'author': 'kimchy',
#     'text': 'Elasticsearch: cool. bonsai cool.',
#     'timestamp': datetime.now(),
# }

# resp = es.index(index="test-index", id=1, document=doc)
# print(resp['result'])

# resp = es.get(index="test-index", id=1)
# print(resp['_source'])

# es.indices.refresh(index="test-index")

# resp = es.search(index="test-index", query={"match_all": {}})
# print("Got %d Hits:" % resp['hits']['total']['value'])
# for hit in resp['hits']['hits']:
#     print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])
