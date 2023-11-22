# from datetime import datetime
from elasticsearch import Elasticsearch
# from elasticsearch import RequestsHttpConnection
# from elasticsearch_dsl import Search, connections
from elasticsearch_dsl import Search, Q, A
from elasticsearch_dsl.query import MultiMatch

from config import settings
from es_model import EsEpisodeTranscript, ScoreMeta, SortableDict


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


async def search_episodes_by_title(show_key: str, qt: str) -> (list, dict):
    print(f'begin search_episodes_by_title for show_key={show_key} qt={qt}')

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

    q = Q('bool', must=[Q('match', title=qt)])

    s = s.filter('term', show_key=show_key)
    s = s.query(q)
    s = s.highlight('title')

    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    raw_query = s.to_dict()

    s = s.execute()

    # print('*************************************************')
    # print(f'response.to_dict()={s.to_dict()}')
    # print('*************************************************')

    for hit in s.hits.hits:
        episode = hit._source
        episode['score'] = hit._score
        # episode['score_meta'] = ScoreMeta(score=hit._score, agg_score=hit._score, high_child_score=0)
        if 'highlight' in hit:
            episode['title'] = hit['highlight']['title'][0]
        results.append(episode._d_)

    return results, raw_query


async def search_scenes(show_key: str, season: str = None, episode_key: str = None, location: str = None, description: str = None) -> (list, int, dict):
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
            inner_hits={
                'size': 100, 
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

    s = s.source(excludes=['scenes'])

    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    raw_query = s.to_dict()

    s = s.execute()

    # print('*************************************************')
    # print(f'response.to_dict()={s.to_dict()}')
    # print('*************************************************')

    scene_count = 0
    for hit in s.hits.hits:
        episode = hit._source
        episode['score_meta'] = ScoreMeta(score=hit._score, agg_score=hit._score, high_child_score=0)
        if 'highlight' in hit:
            episode['title'] = hit['highlight']['title'][0]

        # initialize and highlight inner_hit scenes using scene_offset as keys
        scene_offset_to_scene = {}
        for scene_hit in hit.inner_hits['scenes'].hits.hits:
            scene_offset = scene_hit._nested.offset
            scene = scene_hit._source
            scene['sequence'] = scene_offset
            scene['score_meta'] = ScoreMeta(score=scene_hit._score, agg_score=scene_hit._score, high_child_score=0)
            if 'highlight' in scene_hit:
                if 'scenes.location' in scene_hit.highlight:
                    scene.location = scene_hit.highlight['scenes.location'][0]
                if 'scenes.description' in scene_hit.highlight:
                    scene.description = scene_hit.highlight['scenes.description'][0]
            scene_offset_to_scene[scene_offset] = scene

        # assemble and score episodes from inner_hit scenes / scene_events stitched together above
        episode.scenes = []
        for scene_offset, scene in scene_offset_to_scene.items():
            episode.scenes.append(scene._d_)
            episode['score_meta'].agg_score += scene['score_meta'].agg_score
            episode['score_meta'].high_child_score = max(scene['score_meta'].agg_score, episode['score_meta'].high_child_score)
            scene_count += 1

        # wrap episode in SortableDict before adding to results
        sortable_episode = SortableDict(episode._d_)
        results.append(sortable_episode)

    # sort results before returning
    results = sorted(results, reverse=True)
    return results, scene_count, raw_query


async def search_scene_events(show_key: str, season: str = None, episode_key: str = None, speaker: str = None, dialog: str = None) -> (list, int, int, dict):
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

    nested_q = Q('nested', path='scenes.scene_events', 
            query=q,
            inner_hits={
                'size': 100, 
                'highlight': {
                    'fields': {
                        'scenes.scene_events.spoken_by': {}, 
                        'scenes.scene_events.dialog': {}
                    }
                }
            })

    s = s.query(nested_q)

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)

    s = s.source(excludes=['scenes.scene_events'])

    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    raw_query = s.to_dict()

    s = s.execute()

    # print('*************************************************')
    # print(f'response.to_dict()={s.to_dict()}')
    # print('*************************************************')

    scene_count = 0
    scene_event_count = 0
    for hit in s.hits.hits:
        episode = hit._source
        episode['score_meta'] = ScoreMeta(score=hit._score, agg_score=hit._score, high_child_score=0)
        orig_scenes = episode.scenes

        scene_offset_to_scene = {}

        # highlight and map inner_hit scene_events to scenes using scene_offset
        for scene_event_hit in hit.inner_hits['scenes.scene_events'].hits.hits:
            scene_offset = scene_event_hit._nested.offset
            scene_event = scene_event_hit._source
            scene_event['sequence'] = scene_event_hit._nested._nested.offset
            scene_event['score'] = scene_event_hit._score
            if 'highlight' in scene_event_hit:
                if 'scenes.scene_events.spoken_by' in scene_event_hit.highlight:
                    scene_event_hit._source.spoken_by = scene_event_hit.highlight['scenes.scene_events.spoken_by'][0]
                if 'scenes.scene_events.dialog' in scene_event_hit.highlight:
                    scene_event_hit._source.dialog = scene_event_hit.highlight['scenes.scene_events.dialog'][0]
                # if 'scenes.scene_events.context_info' in scene_event_hit.highlight:
                #     scene_event_hit._source.context_info = scene_event_hit.highlight['scenes.scene_events.context_info'][0]

            # re-assemble scenes with scene_events by mapping parent scene offset to scene list index position
            if scene_offset not in scene_offset_to_scene:
                scene = orig_scenes[scene_offset]
                scene.scene_events = []
                scene['sequence'] = scene_offset
                scene['score_meta'] = ScoreMeta(score=0, agg_score=0, high_child_score=0)
                scene_offset_to_scene[scene_offset] = scene
            scene = scene_offset_to_scene[scene_offset]

            scene['scene_events'].append(scene_event._d_)
            scene['score_meta'].high_child_score = max(scene_event['score'], scene['score_meta'].high_child_score)
            scene['score_meta'].agg_score += scene_event['score']
            scene_event_count += 1

        # assemble and score episodes from inner_hit scenes / scene_events stitched together above
        episode.scenes = []
        for scene_offset, scene in scene_offset_to_scene.items():
            episode.scenes.append(scene._d_)
            episode['score_meta'].agg_score += scene['score_meta'].agg_score
            episode['score_meta'].high_child_score = max(scene['score_meta'].agg_score, episode['score_meta'].high_child_score)
            scene_count += 1
        # wrap episode in SortableDict before adding to results
        sortable_episode = SortableDict(episode._d_)
        results.append(sortable_episode)

    # sort results before returning
    results = sorted(results, reverse=True)
    return results, scene_count, scene_event_count, raw_query


async def search(show_key: str, season: str = None, episode_key: str = None, qt: str = None) -> (list, int, int, dict):
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

    s = Search(using=es_client, index='transcripts')
    s = s.extra(size=1000)

    results = []

    episode_q = Q('match', title=qt)
    
    scene_fields_q = Q('bool', minimum_should_match=1, should=[
            Q('match', **{'scenes.location': qt}),
            Q('match', **{'scenes.description': qt})])
    
    scenes_q = Q('nested', path='scenes', 
            query=scene_fields_q,
            inner_hits={
                'size': 100, 
                'highlight': {
                    'fields': {
                        'scenes.location': {}, 
                        'scenes.description': {}
                    }
                }
            })
    
    scene_event_fields_q = Q('bool', minimum_should_match=1, should=[
            Q('match', **{'scenes.scene_events.context_info': qt}),
            Q('match', **{'scenes.scene_events.spoken_by': qt}),
            Q('match', **{'scenes.scene_events.dialog': qt})])
    
    scene_events_q = Q('nested', path='scenes.scene_events', 
            query=scene_event_fields_q,
            inner_hits={
                'size': 100, 
                'highlight': {
                    'fields': {
                        'scenes.scene_events.spoken_by': {}, 
                        'scenes.scene_events.dialog': {}, 
                        'scenes.scene_events.context_info': {}
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

    s = s.source(excludes=['scenes.scene_events'])

    print('*************************************************')
    print(f's.to_dict()={s.to_dict()}')
    print('*************************************************')

    raw_query = s.to_dict()

    s = s.execute()

    # print('*************************************************')
    # print(f'response.to_dict()={s.to_dict()}')
    # print('*************************************************')

    scene_count = 0
    scene_event_count = 0
    for hit in s.hits.hits:
        episode = hit._source
        episode['score_meta'] = ScoreMeta(score=hit._score, agg_score=hit._score, high_child_score=0)
        if 'highlight' in hit:
            episode['title'] = hit['highlight']['title'][0]
        orig_scenes = episode.scenes

        # initialize and highlight inner_hit scenes using scene_offset as keys
        scene_offset_to_scene = {}
        for scene_hit in hit.inner_hits['scenes'].hits.hits:
            scene_offset = scene_hit._nested.offset
            scene = scene_hit._source
            scene.scene_events = []
            scene['sequence'] = scene_offset
            scene['score_meta'] = ScoreMeta(score=scene_hit._score, agg_score=scene_hit._score, high_child_score=0)
            if 'highlight' in scene_hit:
                if 'scenes.location' in scene_hit.highlight:
                    scene.location = scene_hit.highlight['scenes.location'][0]
                if 'scenes.description' in scene_hit.highlight:
                    scene.description = scene_hit.highlight['scenes.description'][0]
            # del(scene.scene_events)  # TODO handle this in query?
            scene_offset_to_scene[scene_offset] = scene

        # highlight and map inner_hit scene_events to scenes using scene_offset
        for scene_event_hit in hit.inner_hits['scenes.scene_events'].hits.hits:
            scene_offset = scene_event_hit._nested.offset
            scene_event = scene_event_hit._source
            scene_event['sequence'] = scene_event_hit._nested._nested.offset
            scene_event['score'] = scene_event_hit._score
            if 'highlight' in scene_event_hit:
                if 'scenes.scene_events.spoken_by' in scene_event_hit.highlight:
                    scene_event_hit._source.spoken_by = scene_event_hit.highlight['scenes.scene_events.spoken_by'][0]
                if 'scenes.scene_events.dialog' in scene_event_hit.highlight:
                    scene_event_hit._source.dialog = scene_event_hit.highlight['scenes.scene_events.dialog'][0]
                if 'scenes.scene_events.context_info' in scene_event_hit.highlight:
                    scene_event_hit._source.context_info = scene_event_hit.highlight['scenes.scene_events.context_info'][0]

            # if scene at scene_offset wasn't part of inner_hits, grab from top-level _source and initialize it
            if scene_offset not in scene_offset_to_scene:
                scene = orig_scenes[scene_offset]
                scene.scene_events = []
                scene['sequence'] = scene_offset
                scene['score_meta'] = ScoreMeta(score=0, agg_score=0, high_child_score=0)
                scene_offset_to_scene[scene_offset] = scene
            scene = scene_offset_to_scene[scene_offset]

            scene['scene_events'].append(scene_event._d_)
            scene['score_meta'].high_child_score = max(scene_event['score'], scene['score_meta'].high_child_score)
            scene['score_meta'].agg_score += scene_event['score']
            scene_event_count += 1

        # assemble and score episodes from inner_hit scenes / scene_events stitched together above
        episode.scenes = []
        for scene_offset, scene in scene_offset_to_scene.items():
            episode.scenes.append(scene._d_)
            episode['score_meta'].agg_score += scene['score_meta'].agg_score
            episode['score_meta'].high_child_score = max(scene['score_meta'].agg_score, episode['score_meta'].high_child_score)
            scene_count += 1

        # wrap episode in SortableDict before adding to results
        sortable_episode = SortableDict(episode._d_)
        results.append(sortable_episode)

    # sort results before returning
    results = sorted(results, reverse=True)
    return results, scene_count, scene_event_count, raw_query


async def agg_scenes_by_location(show_key: str, episode_key: str = None, season: str = None) -> (list, dict):
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

    raw_query = s.to_dict()
    
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

    return results, raw_query


async def agg_scene_events_by_speaker(show_key: str, episode_key: str = None, season: str = None) -> (list, dict):
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

    raw_query = s.to_dict()

    s = s.execute()

    for item in s.aggregations.speaker_aggs.by_speaker.buckets:
        results[item.key] = item.doc_count

    return results, raw_query
