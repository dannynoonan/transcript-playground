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


async def search_es_episodes(show_key: str, query_term: str) -> list:
    s = Search(using=es_client, index='transcripts')
    results = []

    # MultiMatch(query=query_term, fields=['title', 'episode_key'])

    # q = Q("multi_match", query=query_term, fields=['title', 'episode_key'])
    q = Q('bool', must=[Q('match', show_key=show_key), Q('multi_match', query=query_term, fields=['title', 'episode_key'])])
    # s.query(q)
    s = s.query(q)
    # s = s.query("multi_match", query=query_term, fields=['title', 'episode_key'])
    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    response = s.execute()

    print('*************************************************')
    print(f'response.to_dict()={response.to_dict()}')
    print('*************************************************')

    for hit in response.hits:
        results.append(hit)
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
