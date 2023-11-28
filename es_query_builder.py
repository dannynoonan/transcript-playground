# from datetime import datetime
# from elasticsearch import Elasticsearch
# from elasticsearch import RequestsHttpConnection
from elasticsearch_dsl import Search, connections, Q, A
from elasticsearch_dsl.query import MoreLikeThis

from config import settings
from es_metadata import STOPWORDS
from es_model import EsEpisodeTranscript


# es_client = Elasticsearch(
#     hosts=[{'host': settings.es_host, 'port': settings.es_port, 'scheme': 'https'}],    
#     basic_auth=(settings.es_user, settings.es_pass),
#     verify_certs=False
#     # connection_class=RequestsHttpConnection
# )

# connections.create_connection(hosts=['http://localhost:9200'], timeout=20)

es_conn = connections.create_connection(hosts=[{'host': settings.es_host, 'port': settings.es_port, 'scheme': 'https'}],
                                        basic_auth=(settings.es_user, settings.es_pass), verify_certs=False, timeout=20)

# connections.configure(
#     default={'hosts': 'http://localhost:9200'},
    # dev={
    #     'hosts': ['http://localhost:9200'],
    #     'sniff_on_start': True
    # }
# )


async def init_mappings():
    # EsEpisodeTranscript.init(using=es_client)
    EsEpisodeTranscript.init()


async def save_es_episode(es_episode: EsEpisodeTranscript) -> None:
    # es_episode.save(using=es_client)
    es_episode.save()
    # persisted_es_episode = EsEpisodeTranscript.get(id=es_episode.meta.id, ignore=404)
    # if persisted_es_episode:
    #     es_episode.update(using=es, doc_as_upsert=True)
    # else:
    #     es_episode.save(using=es)


async def fetch_episode_by_key(show_key: str, episode_key: str) -> dict:
    print(f'begin fetch_episode_by_key for show_key={show_key} episode_key={episode_key}')

    # s = Search(using=es_client, index='transcripts')
    s = Search(index='transcripts')
    s = s.extra(size=1)

    s = s.filter('term', show_key=show_key)
    s = s.filter('term', episode_key=episode_key)
    s = s.source(excludes=['flattened_text'])

    return s


async def search_episodes_by_title(show_key: str, qt: str) -> Search:
    print(f'begin search_episodes_by_title for show_key={show_key} qt={qt}')

    # s = Search(using=es_client, index='transcripts')
    s = Search(index='transcripts')
    s = s.extra(size=1000)

    q = Q('bool', must=[Q('match', title=qt)])

    s = s.filter('term', show_key=show_key)
    s = s.query(q)
    s = s.highlight('title')

    return s


async def search_scenes(show_key: str, season: str = None, episode_key: str = None, location: str = None, description: str = None) -> Search:
    print(f'begin search_scenes for show_key={show_key} season={season} episode_key={episode_key} location={location} description={description}')

    if not (location or description):
        print(f'Warning: unable to execute search_scene_events without at least one scene_event property set (location or description)')
        return None
    
    # s = Search(using=es_client, index='transcripts')
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

    s = s.source(excludes=['flattened_text', 'scenes'])

    return s


async def search_scene_events(show_key: str, season: str = None, episode_key: str = None, speaker: str = None, 
                              dialog: str = None, location: str = None) -> Search:
    print(f'begin search_scene_events for show_key={show_key} season={season} episode_key={episode_key} speaker={speaker} dialog={dialog}')
    
    if not (speaker or dialog):
        print(f'Warning: unable to execute search_scene_events without at least one scene_event property set (speaker or dialog)')
        return []

    # s = Search(using=es_client, index='transcripts')
    s = Search(index='transcripts')
    s = s.extra(size=1000)

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
    # if location:
    #     nested_q = nested_q & Q('nested', path='scenes', 
    #         query=Q('match', **{'scenes.location': location}),
    #         inner_hits={'size': 100})

    s = s.query(nested_q)

    s = s.filter('term', show_key=show_key)
    if episode_key:
        s = s.filter('term', episode_key=episode_key)
    if season:
        s = s.filter('term', season=season)
    # if location:
    #     s = s.filter('nested', path='scenes', query=Q('match', **{'scenes.location': location}))

    s = s.source(excludes=['flattened_text', 'scenes.scene_events'])

    return s


async def search_episodes(show_key: str, season: str = None, episode_key: str = None, qt: str = None) -> Search:
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

    # s = Search(using=es_client, index='transcripts')
    s = Search(index='transcripts')
    s = s.extra(size=1000)

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

    s = s.source(excludes=['flattened_text', 'scenes.scene_events'])

    return s


async def agg_scenes_by_location(show_key: str, season: str = None, episode_key: str = None, speaker: str = None) -> Search:
    print(f'begin agg_scenes_by_location for show_key={show_key} season={season} episode_key={episode_key} speaker={speaker}')

    # s = Search(using=es_client, index='transcripts')
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
            'speaker_match', 'filter', filter={"match": {"scenes.scene_events.spoken_by": speaker}}
        ).bucket(
            'scenes', 'reverse_nested', path='scenes'
        ).bucket(
            'by_location', 'terms', field='scenes.location.keyword', size=100)
    else:
        s.aggs.bucket(
            'scenes', 'nested', path='scenes'
        ).bucket(
            'by_location', 'terms', field='scenes.location.keyword', size=100)

    return s


async def agg_scenes_by_speaker(show_key: str, season: str = None, episode_key: str = None, 
                                location: str = None, other_speaker: str = None) -> Search:
    print(f'begin agg_scenes_by_speaker for show_key={show_key} season={season} episode_key={episode_key} location={location} other_speaker={other_speaker}')

    # s = Search(using=es_client, index='transcripts')
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
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=100
        ).bucket(
            'for_scene', 'reverse_nested', path='scenes'
        )
    elif other_speaker:  # TODO location and other_speaker aren't exclusive of each other, this is just a WIP
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'speaker_match', 'filter', filter={"match": {"scenes.scene_events.spoken_by": other_speaker}}
        ).bucket(
            'for_scene', 'reverse_nested', path='scenes'
        ).bucket(
            'scene_events_2', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=100
        ).bucket(
            'for_scene_2', 'reverse_nested', path='scenes'
        )
    else:
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=100
        ).bucket(
            'for_scene', 'reverse_nested', path='scenes'
        )
    
    return s


async def agg_scene_events_by_speaker(show_key: str, season: str = None, episode_key: str = None, dialog: str = None) -> Search:
    print(f'begin agg_scene_events_by_speaker for show_key={show_key} season={season} episode_key={episode_key} dialog={dialog}')

    # s = Search(using=es_client, index='transcripts')
    s = Search(index='transcripts')
    s = s.extra(size=0)

    # if dialog:
    #     nested_q = Q('nested', path='scenes.scene_events', 
    #             query=Q('match', **{'scenes.scene_events.dialog': dialog}),
    #             inner_hits={'size': 100})

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
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=100)
    else:
        s.aggs.bucket(
            'scene_events', 'nested', path='scenes.scene_events'
        ).bucket(
            'by_speaker', 'terms', field='scenes.scene_events.spoken_by.keyword', size=100)

    return s


async def search_keywords_by_episode(show_key: str, episode_key: str) -> dict:
    print(f'begin calc_word_counts_by_episode for show_key={show_key} episode_key={episode_key}')

    response = es_conn.termvectors(index='transcripts', id=f'{show_key}_{episode_key}', term_statistics='true', field_statistics='true',
                                   fields=['flattened_text'], filter={"max_num_terms": 100, "min_term_freq": 1, "min_doc_freq": 1})
                                #    fields=['scenes.scene_events.dialog'])

    return response


async def search_more_like_this(show_key: str, episode_key: str) -> Search:
    print(f'begin search_more_like_this for show_key={show_key} episode_key={episode_key}')

    s = Search(index='transcripts')
    s = s.extra(size=30)

    s = s.query(MoreLikeThis(like=[{'_index': 'transcripts', '_id': f'{show_key}_{episode_key}'}], fields=['flattened_text'],
                             max_query_terms=75, minimum_should_match='75%', min_term_freq=1, stop_words=STOPWORDS))
    
    s = s.source(excludes=['flattened_text', 'scenes'])
    
    return s
