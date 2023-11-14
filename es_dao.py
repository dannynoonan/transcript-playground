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


async def init_mappings():
    EsEpisodeTranscript.init(using=es_client)


async def save_es_episode(es_episode: EsEpisodeTranscript) -> None:
    es_episode.save(using=es_client)
    # persisted_es_episode = EsEpisodeTranscript.get(id=es_episode.meta.id, ignore=404)
    # if persisted_es_episode:
    #     es_episode.update(using=es, doc_as_upsert=True)
    # else:
    #     es_episode.save(using=es)


async def fetch_episode_by_key(show_key: str, episode_key: str) -> EsEpisodeTranscript:
    print(f'begin fetch_episode_by_key for show_key={show_key} episode_key={episode_key}')

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

    s = s.filter('term', show_key=show_key)
    s = s.filter('term', episode_key=episode_key)
    s = s.execute()

    for hit in s.hits:
        results.append(hit)
    return results


async def search_episodes_by_title(show_key: str, qt: str) -> list:
    print(f'begin search_episodes_by_title for show_key={show_key} qt={qt}')

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

    q = Q('bool', must=[Q('match', title=qt)])

    s = s.filter('term', show_key=show_key)
    s = s.query(q)

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


async def search_scenes_by_location(show_key: str, qt: str) -> list:
    print(f'begin search_scenes_by_location for show_key={show_key} qt={qt}')

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

    # q = Q("match", scenes__location=qt)

    s = s.query('nested', path='scenes', 
         query=Q('match', **{'scenes.location': qt}),
         inner_hits={'size': 100},
    )
    s = s.filter('term', show_key=show_key)

    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    s = s.execute()

    print('*************************************************')
    print(f'response.to_dict()={s.to_dict()}')
    print('*************************************************')

    if s.hits and s.hits.hits:
        for hit in s.hits.hits:
            print('*************************************************')
            print(f'hit.to_dict()={hit.to_dict()}')
            print('*************************************************')
            if hit.inner_hits and hit.inner_hits.scenes and hit.inner_hits.scenes.hits and hit.inner_hits.scenes.hits.hits:
                for scene_hit in hit.inner_hits.scenes.hits.hits:
                    results.append(scene_hit)

    return results


async def search_scene_events_by_speaker(show_key: str, qt: str, fields: list = None) -> list:
    print(f'begin search_scene_events_by_speaker for show_key={show_key} qt={qt}')

    # if not fields:
    #     fields = ['scenes.scene_events.dialogue_spoken_by', 'scenes.scene_events.context_info']

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

    s = s.query('nested', path='scenes.scene_events', 
         query=Q('match', **{'scenes.scene_events.spoken_by': qt}),
         inner_hits={'size': 100},
    )
    s = s.filter('term', show_key=show_key)

    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    s = s.execute()

    print('*************************************************')
    print(f'response.to_dict()={s.to_dict()}')
    print('*************************************************')

    if s.hits and s.hits.hits:
        for hit in s.hits.hits:
            print('*************************************************')
            print(f'hit.to_dict()={hit.to_dict()}')
            print('*************************************************')
            if hit.inner_hits and hit.inner_hits['scenes.scene_events'] and hit.inner_hits['scenes.scene_events'].hits and hit.inner_hits['scenes.scene_events'].hits.hits:
                for scene_event_hit in hit.inner_hits['scenes.scene_events'].hits.hits:
                    results.append(scene_event_hit)

    return results


async def agg_episodes_by_location(show_key: str) -> list:
    print(f'begin agg_episodes_by_location for show_key={show_key}')

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = {}

    q = Q('bool', must=[Q('match', show_key=show_key)])
    s = s.query(q)
    s.aggs.bucket(f'by_location', 'terms', field='scenes.location', size=1000)

    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    s = s.execute()

    print('*************************************************')
    print(f'response.to_dict()={s.to_dict()}')
    print('*************************************************')


    print('*************************************************')
    print(f's.aggregations.by_location={s.aggregations.by_location}')
    print('*************************************************')
    print(f's.hits.total={s.hits.total}')
    print('*************************************************')
    print(f's.aggregations.by_location.buckets={s.aggregations.by_location.buckets}')
    print('*************************************************')

    for item in s.aggregations.by_location.buckets:
        results[item.key] = item.doc_count

    return results


async def agg_episodes_by_character(show_key: str) -> list:
    print(f'begin agg_episodes_by_character for show_key={show_key}')

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
