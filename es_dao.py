# from datetime import datetime
from elasticsearch import Elasticsearch
# from elasticsearch import RequestsHttpConnection
# from elasticsearch_dsl import Search, connections
from elasticsearch_dsl import Search, Q, A
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


# async def search_scenes_by_location(show_key: str, qt: str, episode_key: str = None, season: str = None) -> list:
#     print(f'begin search_scenes_by_location for show_key={show_key} qt={qt}')

#     s = Search(using=es_client, index='transcripts')
#     s = s.extra(size=1000)

#     results = []

#     # q = Q("match", scenes__location=qt)

#     s = s.query('nested', path='scenes', 
#             query=Q('match', **{'scenes.location': qt}),
#             inner_hits={'size': 100},
#     )
#     s = s.filter('term', show_key=show_key)
#     if episode_key:
#         s = s.filter('term', episode_key=episode_key)
#     if season:
#         s = s.filter('term', season=season)

#     print('*************************************************')
#     print(f's.to_dict()={s.to_dict()}')
#     print('*************************************************')

#     s = s.execute()

#     # print('*************************************************')
#     # print(f'response.to_dict()={s.to_dict()}')
#     # print('*************************************************')

#     if s.hits and s.hits.hits:
#         for hit in s.hits.hits:
#             # print('*************************************************')
#             # print(f'hit.to_dict()={hit.to_dict()}')
#             # print('*************************************************')
#             if hit.inner_hits and hit.inner_hits.scenes and hit.inner_hits.scenes.hits and hit.inner_hits.scenes.hits.hits:
#                 for scene_hit in hit.inner_hits.scenes.hits.hits:
#                     results.append(scene_hit)

#     return results


async def search_scenes(show_key: str, season: str = None, episode_key: str = None, location: str = None, description: str = None) -> (list, int):
    print(f'begin search_scenes for show_key={show_key} season={season} episode_key={episode_key} location={location} description={description}')

    if not (location or description):
        print(f'Warning: unable to execute search_scene_events without at least one scene_event property set (location or description)')
        return [], 0
    
    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

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
            inner_hits={'size': 100, 'highlight': {'fields': {'scenes.location': {}, 'scenes.description': {}}}},
    )

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    s = s.source(excludes=['scenes'])

    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    s = s.execute()

    # print('*************************************************')
    # print(f'response.to_dict()={s.to_dict()}')
    # print('*************************************************')

    # hits and inner_hits come back fragmented, reassemble them and track inner_hit count
    scene_count = 0
    if s.hits and s.hits.hits:
        for hit in s.hits.hits:
            scene_hits = []
            for scene_hit in hit.inner_hits.scenes.hits.hits:
                if 'highlight' in scene_hit and 'scenes.location' in scene_hit.highlight:
                    scene_hit._source.location = scene_hit.highlight["scenes.location"]._l_[0]
                if 'highlight' in scene_hit and 'scenes.description' in scene_hit.highlight:
                    scene_hit._source.description = scene_hit.highlight["scenes.description"]._l_[0]
                scene_hits.append(scene_hit._source._d_)
                scene_count += 1
            hit.inner_hits = scene_hits
            results.append(hit._d_)

    return results, scene_count


# async def search_scene_events_by_speaker(show_key: str, qt: str, episode_key: str = None, season: str = None) -> list:
#     print(f'begin search_scene_events_by_speaker for show_key={show_key} qt={qt}')

#     # if not fields:
#     #     fields = ['scenes.scene_events.dialogue_spoken_by', 'scenes.scene_events.context_info']

#     s = Search(using=es_client, index='transcripts')
#     s = s.extra(size=1000)

#     results = []

#     s = s.query('nested', path='scenes.scene_events', 
#             query=Q('match', **{'scenes.scene_events.spoken_by': qt}),
#             inner_hits={'size': 100},
#     )
#     s = s.filter('term', show_key=show_key)
#     if episode_key:
#         s = s.filter('term', episode_key=episode_key)
#     if season:
#         s = s.filter('term', season=season)

#     print('*************************************************')
#     print(f's.to_dict()={s.to_dict()}')
#     print('*************************************************')

#     s = s.execute()

#     # print('*************************************************')
#     # print(f'response.to_dict()={s.to_dict()}')
#     # print('*************************************************')

#     if s.hits and s.hits.hits:
#         for hit in s.hits.hits:
#             # print('*************************************************')
#             # print(f'hit.to_dict()={hit.to_dict()}')
#             # print('*************************************************')
#             if hit.inner_hits and hit.inner_hits['scenes.scene_events'] and hit.inner_hits['scenes.scene_events'].hits and hit.inner_hits['scenes.scene_events'].hits.hits:
#                 for scene_event_hit in hit.inner_hits['scenes.scene_events'].hits.hits:
#                     results.append(scene_event_hit)

#     return results


async def search_scene_events(show_key: str, season: str = None, episode_key: str = None, speaker: str = None, dialog: str = None) -> (list, int):
    print(f'begin search_scene_events for show_key={show_key} season={season} episode_key={episode_key} speaker={speaker} dialog={dialog}')
    if not (speaker or dialog):
        print(f'Warning: unable to execute search_scene_events without at least one scene_event property set (speaker or dialog)')
        return []

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

    speaker_q = Q('match', **{'scenes.scene_events.spoken_by': speaker})
    dialog_q = Q('match', **{'scenes.scene_events.dialog': dialog})

    q = None
    if speaker:
        q = speaker_q
        if dialog:
            q = q & dialog_q
    else:
        q = dialog_q

    s = s.query('nested', path='scenes.scene_events', 
            query=q,
            inner_hits={'size': 100, 'highlight': {'fields': {'scenes.scene_events.spoken_by': {}, 'scenes.scene_events.dialog': {}}}},
    )

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    s = s.source(excludes=['scenes.scene_events'])

    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    s = s.execute()

    # print('*************************************************')
    # print(f'response.to_dict()={s.to_dict()}')
    # print('*************************************************')

    # hits and inner_hits come back fragmented, reassemble them and track inner_hit count
    scene_count = 0
    scene_event_count = 0
    if s.hits and s.hits.hits:
        for hit in s.hits.hits:
            scenes = hit._source.scenes
            scene_offsets_to_scene_event_lists = {}
            # map scene offsets to scene_events
            for scene_event_hit in hit.inner_hits['scenes.scene_events'].hits.hits:
                scene_offset = scene_event_hit._nested.offset
                scene_event = scene_event_hit._d_['_source']
                scene_event['_score'] = scene_event_hit._d_['_score']
                if 'highlight' in scene_event_hit and 'scenes.scene_events.spoken_by' in scene_event_hit.highlight:
                    scene_event_hit._source.spoken_by = scene_event_hit.highlight["scenes.scene_events.spoken_by"]._l_[0]
                if 'highlight' in scene_event_hit and 'scenes.scene_events.dialog' in scene_event_hit.highlight:
                    scene_event_hit._source.dialog = scene_event_hit.highlight["scenes.scene_events.dialog"]._l_[0]
                if scene_offset in scene_offsets_to_scene_event_lists:
                    scene_offsets_to_scene_event_lists[scene_offset].append(scene_event)
                else:
                    scene_offsets_to_scene_event_lists[scene_offset] = [scene_event]
                    scene_count += 1
                scene_event_count += 1
                
            # re-assemble scenes with scene_events by mapping parent scene offset to scene list index position
            scene_inner_hits = []
            for scene_offset, scene_events in scene_offsets_to_scene_event_lists.items():
                scene_to_scene_events = scenes[scene_offset]
                scene_to_scene_events['sequence'] = scene_offset
                scene_to_scene_events['scene_events'] = scene_events
                scene_inner_hits.append(scene_to_scene_events._d_)
            # re-assemble episodes with scenes 
            hit.inner_hits = scene_inner_hits
            del(hit._source['scenes'])
            results.append(hit._d_)

    return results, scene_event_count


async def search(show_key: str, season: str = None, episode_key: str = None, qt: str = None):
    '''
    qt can be: 
        episode.title
        episode.scene.location
        episode.scene.description
        episode.scene.scene_event.context_info
        episode.scene.scene_event.spoken_by
        episode.scene.scene_event.dialog
    '''

    results = []
    return results


async def agg_scenes_by_location(show_key: str, episode_key: str = None, season: str = None) -> list:
    print(f'begin agg_scenes_by_location for show_key={show_key}')

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=0)

    results = {}

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    s.aggs.bucket('location_aggs', 'nested', path='scenes').bucket('by_location', 'terms', field='scenes.location', size=100)

    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    s = s.execute()

    # print('*************************************************')
    # print(f's.aggregations.location_aggs={s.aggregations.location_aggs}')
    # print('*************************************************')
    # print(f's.aggregations.location_aggs.by_location={s.aggregations.location_aggs.by_location}')
    # print('*************************************************')
    # print(f's.hits.total={s.hits.total}')
    # print('*************************************************')
    # print(f's.aggregations.location_aggs.by_location.buckets={s.aggregations.location_aggs.by_location.buckets}')
    # print('*************************************************')

    for item in s.aggregations.location_aggs.by_location.buckets:
        results[item.key] = item.doc_count

    return results


async def agg_scene_events_by_speaker(show_key: str, episode_key: str = None, season: str = None) -> list:
    print(f'begin agg_scene_events_by_speaker for show_key={show_key}')

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=0)

    results = {}

    # q = Q('bool', must=[Q('match', show_key=show_key)])
    # s = s.query(q)
    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    s.aggs.bucket('speaker_aggs', 'nested', path='scenes.scene_events').bucket('by_speaker', 'terms', field='scenes.scene_events.spoken_by', size=100)

    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    s = s.execute()

    for item in s.aggregations.speaker_aggs.by_speaker.buckets:
        results[item.key] = item.doc_count

    return results
