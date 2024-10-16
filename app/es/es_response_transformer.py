from elasticsearch_dsl import Search
from operator import itemgetter

from app.es.es_metadata import STOPWORDS


def return_episode_by_key(s: Search) -> dict:
    # print(f'begin return_episode_by_key for s.to_dict()={s.to_dict()}')

    s = s.execute()

    for hit in s.hits:
        return hit._d_
    

def return_doc_ids(s: Search) -> list:
    # print(f'begin return_doc_ids for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    for hit in s.hits.hits:
        results.append(hit['_id'])
    
    return results


def return_speakers(s: Search) -> list:
    # print(f'begin return_speakers for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    for hit in s.hits.hits:
        results.append(hit._source._d_)
    
    return results


def return_speaker_seasons(s: Search) -> list:
    # print(f'begin return_speaker_seasons for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    for hit in s.hits.hits:
        results.append(hit._source._d_)

    return results


def return_speaker_episodes(s: Search) -> list:
    # print(f'begin return_speaker_episodes for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    for hit in s.hits.hits:
        results.append(hit._source._d_)

    return results


def return_topic(s: Search) -> dict:
    # print(f'begin return_topic for s.to_dict()={s.to_dict()}')

    s = s.execute()

    for hit in s.hits:
        return hit._d_
    

def return_topics(s: Search) -> dict:
    # print(f'begin return_topics for s.to_dict()={s.to_dict()}')

    s = s.execute()

    topics = []

    for hit in s.hits.hits:
        topics.append(hit._source._d_)

    return topics


def return_topics_by_episode(s: Search) -> dict:
    # print(f'begin return_topics_by_episode for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}

    for hit in s.hits.hits:
        if hit._source.episode_key not in results:
            results[hit._source.episode_key] = []
        results[hit._source.episode_key].append(hit._source._d_)
    
    return results


def return_topics_by_season(s: Search) -> dict:
    # print(f'begin return_topics_by_season for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}

    for hit in s.hits.hits:
        if hit._source.season not in results:
            results[hit._source.season] = []
        results[hit._source.season].append(hit._source._d_)
    
    return results


def return_topics_by_speaker(s: Search) -> dict:
    # print(f'begin return_topics_by_speaker for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}

    for hit in s.hits.hits:
        if hit._source.speaker not in results:
            results[hit._source.speaker] = []
        results[hit._source.speaker].append(hit._source._d_)
    
    return results
    

def return_episodes_by_title(s: Search) -> list:
    # print(f'begin return_episodes_by_title for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    for hit in s.hits.hits:
        episode = hit._source
        episode['score'] = hit._score
        if 'highlight' in hit:
            episode['title'] = hit['highlight']['title'][0]
        results.append(episode._d_)

    return results


def return_scenes(s: Search) -> tuple[list, int]:
    # print(f'begin return_scenes for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    scene_count = 0
    for hit in s.hits.hits:
        episode = hit._source
        episode['score'] = hit._score
        episode['agg_score'] = hit._score
        episode['high_child_score'] = 0
        if 'highlight' in hit:
            episode['title'] = hit['highlight']['title'][0]

        # initialize and highlight inner_hit scenes using scene_offset as keys
        scene_offset_to_scene = {}
        for scene_hit in hit.inner_hits['scenes'].hits.hits:
            scene_offset = scene_hit._nested.offset
            scene = scene_hit._source
            scene['sequence'] = scene_offset
            scene['score'] = scene_hit._score
            scene['agg_score'] = scene_hit._score
            scene['high_child_score'] = 0
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
            episode['agg_score'] += scene['agg_score']
            episode['high_child_score'] = max(scene['agg_score'], episode['high_child_score'])
            scene_count += 1
        results.append(episode._d_)

    # sort results before returning
    results = sorted(results, key=itemgetter('agg_score'), reverse=True)

    return results, scene_count


def return_narrative_sequences(s: Search) -> list:
    # print(f'begin return_narrative_sequences for s.to_dict()={s.to_dict()}')

    s = s.execute()

    narrative_sequences = []

    for hit in s.hits.hits:
        narrative_sequence = hit._source._d_
        # manually re-order clusters by probability desc, since non-nested 'Object' field doesn't support ordering
        if 'cluster_memberships' in narrative_sequence:
            clusters = narrative_sequence['cluster_memberships']
            narrative_sequence['cluster_memberships'] = sorted(clusters, key=itemgetter('probability'), reverse=True)
        narrative_sequences.append(narrative_sequence)

    return narrative_sequences


def return_flattened_scenes(s: Search, include_speakers: bool = False, include_context: bool = False, line_breaks: bool = False) -> dict:
    # print(f'begin return_flattened_scenes for s.to_dict()={s.to_dict()}')

    s = s.execute()

    flattened_scenes = []

    split_str = ' '
    if line_breaks:
        split_str = '\n\n'

    for hit in s.hits:
        episode = hit._d_
        if 'scenes' not in episode:
            continue
        for scene in episode['scenes']:
            scene_event_dialog = []
            # NOTE: to preserve scene index positions, append scene even if it lacks dialog content
            if 'scene_events' not in scene:
                flattened_scenes.append('')
                continue
            for scene_event in scene['scene_events']:
                scene_event_elements = []
                if include_context and 'context_info' in scene_event and not scene_event['context_info'] == 'OC':
                    scene_event_elements.append(f"[{scene_event['context_info']}]")
                if include_speakers and 'spoken_by' in scene_event:
                    scene_event_elements.append(f"{scene_event['spoken_by']}:")
                if 'dialog' in scene_event:
                    scene_event_elements.append(scene_event['dialog'])
                if scene_event_elements: 
                    scene_event_dialog.append(' '.join(scene_event_elements))
            flattened_scenes.append(f'{split_str}'.join(scene_event_dialog))
    
    return flattened_scenes


def return_scene_events(s: Search, location: str = None) -> tuple[list, int, int]:
    print(f'begin return_scene_events for location={location} s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    scene_count = 0
    scene_event_count = 0
    for hit in s.hits.hits:
        episode = hit._source
        episode['score'] = hit._score
        episode['agg_score'] = hit._score
        episode['high_child_score'] = 0
        episode['scene_event_count'] = 0
        episode['word_count'] = 0
        orig_scenes = episode.scenes

        scene_offset_to_scene = {}

        # highlight and map inner_hit scene_events to scenes using scene_offset
        for scene_event_hit in hit.inner_hits['scenes.scene_events'].hits.hits:
            scene_offset = scene_event_hit._nested.offset
            # NOTE hack to filter on location after es payload returned, until I find a way to cross-reference nested documents
            # or punt and enable `include_in_root`
            if location and location not in orig_scenes[scene_offset]['location']:
                # print(f'location={location} != orig_scenes[scene_offset]={orig_scenes[scene_offset]}')
                continue
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
                scene['score'] = 0
                scene['agg_score'] = 0
                scene['high_child_score'] = 0
                scene_offset_to_scene[scene_offset] = scene
            scene = scene_offset_to_scene[scene_offset]

            scene['scene_events'].append(scene_event._d_)
            scene['high_child_score'] = max(scene_event['score'], scene['high_child_score'])
            scene['agg_score'] += scene_event['score']
            episode['scene_event_count'] += 1
            # NOTE to be consistent, word_count would match the output of agg_dialog_word_counts (if that endpoint had a 'group by episode' option, which it doesn't)
            if 'dialog' in scene_event._d_:
                episode['word_count'] += len(scene_event._d_['dialog'].split(' '))
            scene_event_count += 1

        # NOTE follow-up to location-filter hack above, if all scenes have been filtered then skip episode 
        if len(scene_offset_to_scene) == 0:
            # print(f'dropping episode_key={episode["episode_key"]}, no scenes remain after location filtering')
            continue

        # assemble and score episodes from inner_hit scenes / scene_events stitched together above
        episode.scenes = []
        # sort scenes by sequence
        scene_offset_to_scene = dict(sorted(scene_offset_to_scene.items()))
        for scene_offset, scene in scene_offset_to_scene.items():
            # sort scene_events by sequence
            sorted_scene_events = sorted(scene._d_['scene_events'], key=itemgetter('sequence'))
            scene._d_['scene_events'] = sorted_scene_events
            # highlight scene.location match here for now (a little ugly)
            if location:
                scene._d_['location'] = scene._d_['location'].replace(location, f'<em>{location}</em>')
            episode.scenes.append(scene._d_)
            episode['agg_score'] += scene['agg_score']
            episode['high_child_score'] = max(scene['agg_score'], episode['high_child_score'])
            scene_count += 1
        results.append(episode._d_)

    # sort results before returning
    results = sorted(results, key=itemgetter('agg_score'), reverse=True)

    return results, scene_count, scene_event_count


def return_scene_events_multi_speaker(s: Search, speakers: str, location: str = None, intersection: bool = False) -> tuple[list, int, int]:
    # print(f'begin return_scene_events_multi_speaker for speakers={speakers} location={location} s.to_dict()={s.to_dict()}')

    s = s.execute()

    # NOTE `speakers` are actually `inner_hit_names`, if I end up making this more generic
    speakers = speakers.split(',')

    results = []

    scene_count = 0
    scene_event_count = 0
    for hit in s.hits.hits:
        episode = hit._source
        episode['score'] = hit._score
        episode['agg_score'] = hit._score
        episode['high_child_score'] = 0
        orig_scenes = episode.scenes

        scene_offset_to_scene = {}

        for speaker in speakers:
            # highlight and map inner_hit scene_events to scenes using scene_offset
            for scene_event_hit in hit.inner_hits[speaker].hits.hits:
                scene_offset = scene_event_hit._nested.offset
                # NOTE hack to filter on location after es payload returned, until I find a way to cross-reference nested documents
                # or punt and enable `include_in_root`
                if location and orig_scenes[scene_offset]['location'] != location:
                    # print(f'location={location} != orig_scenes[scene_offset]={orig_scenes[scene_offset]}')
                    continue
                scene_event = scene_event_hit._source
                scene_event['sequence'] = scene_event_hit._nested._nested.offset
                scene_event['score'] = scene_event_hit._score
                if 'highlight' in scene_event_hit and 'scenes.scene_events.spoken_by' in scene_event_hit.highlight:
                    scene_event_hit._source.spoken_by = scene_event_hit.highlight['scenes.scene_events.spoken_by'][0]

                # re-assemble scenes with scene_events by mapping parent scene offset to scene list index position
                if scene_offset not in scene_offset_to_scene:
                    scene = orig_scenes[scene_offset]
                    scene.scene_events = []
                    scene['sequence'] = scene_offset
                    scene['score'] = 0
                    scene['agg_score'] = 0
                    scene['high_child_score'] = 0
                    scene_offset_to_scene[scene_offset] = scene
                scene = scene_offset_to_scene[scene_offset]

                scene['scene_events'].append(scene_event._d_)
                scene['high_child_score'] = max(scene_event['score'], scene['high_child_score'])
                scene['agg_score'] += scene_event['score']
                scene_event_count += 1

        # NOTE follow-up to location-filter hack above, if all scenes have been filtered then skip episode 
        if len(scene_offset_to_scene) == 0:
            # print(f'dropping episode_key={episode["episode_key"]}, no scenes remain after location filtering')
            continue

        # assemble and score episodes from inner_hit scenes / scene_events stitched together above
        episode.scenes = []
        # sort scenes by sequence
        scene_offset_to_scene = dict(sorted(scene_offset_to_scene.items()))
        for scene_offset, scene in scene_offset_to_scene.items():
            # sort scene_events by sequence
            sorted_scene_events = sorted(scene._d_['scene_events'], key=itemgetter('sequence'))
            scene._d_['scene_events'] = sorted_scene_events
            # if intersection=True then all speakers must be present for scene to be added
            # NOTE ideally this would be added at query level, but feels tricky and not high priority
            if intersection:
                speakers_in_scene = set()
                for sse in sorted_scene_events:
                    speakers_in_scene.add(sse['spoken_by'])
                if len(speakers_in_scene) < len(speakers):
                    # print(f'dropping scene at scene_offset={scene_offset}, speakers_in_scene={speakers_in_scene} is a subset of speakers={speakers}')
                    continue
            # highlight scene.location match here for now (a little ugly)
            if location:
                scene._d_['location'] = scene._d_['location'].replace(location, f'<em>{location}</em>')
            episode.scenes.append(scene._d_)
            episode['agg_score'] += scene['agg_score']
            episode['high_child_score'] = max(scene['agg_score'], episode['high_child_score'])
            scene_count += 1
        results.append(episode._d_)

    # sort results before returning
    results = sorted(results, key=itemgetter('agg_score'), reverse=True)

    return results, scene_count, scene_event_count


def return_seasons(s: Search) -> list:
    # print(f'begin return_seasons for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    for item in s.aggregations.by_season.buckets:
        results.append(item.key)

    return results


def return_seasons_by_speaker(s: Search, agg_season_count: str, location: str = None) -> list:
    # print(f'begin return_seasons_by_speaker for location={location} s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}
    results['_ALL_'] = agg_season_count

    if location:
        pass  # TODO copied from return_episodes_by_speaker
        # for item in s.aggregations.scenes.location_match.scene_events.by_speaker.buckets:
        #     results[item.key] = item.for_episode.doc_count
    else:
        for item in s.aggregations.scene_events.by_speaker.buckets:
            results[item.key] = len(item.by_season.season.buckets)

    # reverse nesting throws off sorting, so sort results by value
    sorted_results_list = sorted(results.items(), key=lambda x:x[1], reverse=True)
    results = {}
    for speaker, count in sorted_results_list:
        results[speaker] = count

    return results


def return_seasons_by_location(s: Search, agg_season_count: str) -> list:
    # print(f'begin return_seasons_by_speaker s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}
    results['_ALL_'] = agg_season_count

    for item in s.aggregations.scenes.by_location.buckets:
        results[item.key] = len(item.by_season.season.buckets)

    # reverse nesting throws off sorting, so sort results by value
    sorted_results_list = sorted(results.items(), key=lambda x:x[1], reverse=True)
    results = {}
    for speaker, count in sorted_results_list:
        results[speaker] = count

    return results


def return_season_count(s: Search) -> int:
    # print(f'begin return_episode_count for s.to_dict()={s.to_dict()}')

    s = s.execute()

    return len(s.aggregations.by_season.buckets)


def return_episodes(s: Search) -> tuple[list, int, int]:
    # print(f'begin return_episodes for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    scene_count = 0
    scene_event_count = 0
    for hit in s.hits.hits:
        episode = hit._source
        episode['score'] = hit._score
        episode['agg_score'] = hit._score
        episode['high_child_score'] = 0
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
            scene['score'] = scene_hit._score
            scene['agg_score'] = scene_hit._score
            scene['high_child_score'] = 0
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
                scene['score'] = 0
                scene['agg_score'] = 0
                scene['high_child_score'] = 0
                scene_offset_to_scene[scene_offset] = scene
            scene = scene_offset_to_scene[scene_offset]

            scene['scene_events'].append(scene_event._d_)
            scene['high_child_score'] = max(scene_event['score'], scene['high_child_score'])
            scene['agg_score'] += scene_event['score']
            scene_event_count += 1

        # assemble and score episodes from inner_hit scenes / scene_events stitched together above
        episode.scenes = []
        # sort scenes by sequence
        scene_offset_to_scene = dict(sorted(scene_offset_to_scene.items()))
        for scene_offset, scene in scene_offset_to_scene.items():
            # sort scene_events by sequence
            sorted_scene_events = sorted(scene._d_['scene_events'], key=itemgetter('sequence'))
            scene._d_['scene_events'] = sorted_scene_events
            episode.scenes.append(scene._d_)
            episode['agg_score'] += scene['agg_score']
            episode['high_child_score'] = max(scene['agg_score'], episode['high_child_score'])
            scene_count += 1
        results.append(episode._d_)

    # sort results before returning
    results = sorted(results, key=itemgetter('agg_score'), reverse=True)

    return results, scene_count, scene_event_count


def return_simple_episodes(s: Search) -> list:
    # print(f'begin return_simple_episodes for s.to_dict()={s.to_dict()}')

    s = s.execute()

    episodes = []

    for hit in s.hits.hits:
        episodes.append(hit._source._d_)

    return episodes


def return_episodes_by_season(s: Search) -> dict:
    # print(f'begin return_episodes_by_season for s.to_dict()={s.to_dict()}')

    s = s.execute()

    seasons_to_episodes = {}

    for hit in s.hits.hits:
        episode = hit._source
        if episode['season'] in seasons_to_episodes:
            seasons_to_episodes[episode['season']].append(episode._d_)
        else:
            seasons_to_episodes[episode['season']] = [episode._d_]

    # sort results by season and sequence_in_season
    # sorted_seasons_to_episodes = dict(sorted(seasons_to_episodes.items()))
    # for season, episodes in sorted_seasons_to_episodes.items():
    #     episodes = sorted(episodes, itemgetter='sequence_in_season')

    return seasons_to_episodes


def return_all_episode_relations(s: Search) -> dict:
    # print(f'begin return_all_episode_relations for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    for hit in s.hits.hits:
        results.append(hit._source._d_)

    return results


def return_episode_count(s: Search) -> int:
    # print(f'begin return_episode_count for s.to_dict()={s.to_dict()}')

    s = s.execute()

    return int(s.hits.total.value)


def return_episodes_by_speaker(s: Search, agg_episode_count: str, location: str = None, other_speaker: str = None) -> list:
    # print(f'begin return_episodes_by_speaker for location={location} other_speaker={other_speaker} s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}
    results['_ALL_'] = agg_episode_count

    if location:
        for item in s.aggregations.scenes.location_match.scene_events.by_speaker.buckets:
            results[item.key] = item.for_episode.doc_count
    elif other_speaker:
        for item in s.aggregations.scene_events.speaker_match.for_scene.scene_events_2.by_speaker.buckets:
            results[item.key] = item.for_episode.doc_count
    else:
        for item in s.aggregations.scene_events.by_speaker.buckets:
            results[item.key] = item.for_episode.doc_count

    # reverse nesting throws off sorting, so sort results by value
    sorted_results_list = sorted(results.items(), key=lambda x:x[1], reverse=True)
    results = {}
    for speaker, count in sorted_results_list:
        results[speaker] = count

    return results


def return_episodes_by_location(s: Search, agg_episode_count: str) -> list:
    # print(f'begin return_episodes_by_speaker s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}
    results['_ALL_'] = agg_episode_count

    for item in s.aggregations.scenes.by_location.buckets:
        results[item.key] = item.by_episode.doc_count

    # reverse nesting throws off sorting, so sort results by value
    sorted_results_list = sorted(results.items(), key=lambda x:x[1], reverse=True)
    results = {}
    for speaker, count in sorted_results_list:
        results[speaker] = count

    return results


def return_scene_count(s: Search) -> int:
    # print(f'begin return_scene_count for s.to_dict()={s.to_dict()}')

    s = s.execute()

    return int(s.aggregations.scene_count.value)


def return_scenes_by_location(s: Search, speaker: str = None) -> list:
    # print(f'begin return_scenes_by_location for speaker={speaker} s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}
    results['_ALL_'] = 0

    if speaker:
        for item in s.aggregations.scene_events.speaker_match.scenes.by_location.buckets:
            results['_ALL_'] += item.doc_count
            results[item.key] = item.doc_count
    else:
        for item in s.aggregations.scenes.by_location.buckets:
            results['_ALL_'] += item.doc_count
            results[item.key] = item.doc_count

    return results


def return_scenes_by_speaker(s: Search, agg_scene_count: str, location: str = None, other_speaker: str = None) -> list:
    # print(f'begin return_scenes_by_speaker for location={location} other_speaker={other_speaker} s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}
    results['_ALL_'] = agg_scene_count

    if location:
        for item in s.aggregations.scenes.location_match.scene_events.by_speaker.buckets:
            results[item.key] = item.for_scene.doc_count
    elif other_speaker:
        for item in s.aggregations.scene_events.speaker_match.for_scene.scene_events_2.by_speaker.buckets:
            results[item.key] = item.for_scene_2.doc_count
    else:
        for item in s.aggregations.scene_events.by_speaker.buckets:
            results[item.key] = item.for_scene.doc_count

    # reverse nesting throws off sorting, so sort results by value
    sorted_results_list = sorted(results.items(), key=lambda x:x[1], reverse=True)
    results = {}
    for speaker, count in sorted_results_list:
        results[speaker] = count

    return results


def return_scene_events_by_speaker(s: Search, dialog: str = None) -> list:
    # print(f'begin return_scene_events_by_speaker for dialog={dialog} s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}
    results['_ALL_'] = 0

    if dialog:
        for item in s.aggregations.scene_events.dialog_match.by_speaker.buckets:
            results['_ALL_'] += item.doc_count
            results[item.key] = item.doc_count
    else:
        for item in s.aggregations.scene_events.by_speaker.buckets:
            results['_ALL_'] += item.doc_count
            results[item.key] = item.doc_count

    return results


def return_dialog_word_counts(s: Search, speaker: str = None) -> list:
    # print(f'begin return_dialog_word_counts s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}
    results['_ALL_'] = 0

    if speaker:
        results = {speaker: s.aggregations.scene_events.speaker_match.word_count._d_['value']}
    else:
        for item in s.aggregations.scene_events.by_speaker.buckets:
            results['_ALL_'] += item.word_count.value
            results[item.key] = item.word_count.value

        # default sorting is by doc_count, we want to sort by word_count
        sorted_results_list = sorted(results.items(), key=lambda x:x[1], reverse=True)
        results = {}
        for speaker, word_count in sorted_results_list:
            results[speaker] = word_count

    return results


def return_numeric_distrib_into_percentiles(s: Search, numeric_field: str) -> dict:
    # print(f'begin return_numeric_distrib_into_percentiles s.to_dict()={s.to_dict()} numeric_field={numeric_field}')

    s = s.execute()

    return s.aggregations[f'{numeric_field}_slices'].values._d_


def return_keywords_by_episode(query_response: dict, exclude_terms: bool = False) -> list:
    # print(f'begin return_keywords_by_episode for len(query_response)={len(query_response)} exclude_terms={exclude_terms}')

    results = []

    if not query_response["term_vectors"]:
        return results    

    # for term, data in query_response['term_vectors']['scenes.scene_events.dialog']['terms'].items():
    for term, data in query_response['term_vectors']['flattened_text']['terms'].items():
        if term in STOPWORDS or term.upper() in exclude_terms:
            continue
        term_dict = {}
        term_dict['term'] = term
        term_dict['term_freq'] = data['term_freq']
        term_dict['doc_freq'] = data['doc_freq']
        term_dict['ttf'] = data['ttf']
        term_dict['score'] = data['score']
        results.append(term_dict)

    # sort results before returning
    results = sorted(results, key=itemgetter('score'), reverse=True)

    return results


def return_keywords_by_corpus(query_response: dict, exclude_terms: bool = False) -> list:
    # print(f'begin return_keywords_by_corpus for len(query_response)={len(query_response)} exclude_terms={exclude_terms}')

    # TODO if this is a partial corpus (e.g. single season) then need to aggregate term-freq-for-corpus ad hoc rather than use the 'ttf' value 
    results = []

    if not query_response["docs"]:
        return results
        
    all_term_dicts = {}
    for doc in query_response["docs"]:
        for term, data in doc['term_vectors']['flattened_text']['terms'].items():
            if term in all_term_dicts or term in STOPWORDS or term.upper() in exclude_terms:
                continue
            term_dict = {}
            term_dict['term'] = term
            term_dict['doc_freq'] = data['doc_freq']
            term_dict['ttf'] = data['ttf']
            all_term_dicts[term] = term_dict

    # sort results before returning
    results = sorted(all_term_dicts.values(), key=itemgetter('ttf'), reverse=True)

    return results


def return_more_like_this(s: Search) -> list:
    # print(f'begin return_more_like_this for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    for hit in s.hits.hits:
        episode = hit._source
        episode['score'] = hit._score
        results.append(episode._d_)

    return results


def return_vector_search(es_response: dict) -> list:
    # print(f'begin return_vector_search')

    results = []
    
    rank = 1
    for hit in es_response['hits']['hits']:
        match = hit['_source']
        match['score'] = hit['_score'] * 100
        match['rank'] = rank
        rank += 1
        results.append(match)

    return results


def return_topic_vector_search(es_response: dict) -> list:
    # print(f'begin return_topic_vector_search')

    # TODO currently exactly the same as `return_vector_search`, maybe we don't need both

    results = []
    
    rank = 1
    for hit in es_response['hits']['hits']:
        topic = hit['_source']
        topic['score'] = hit['_score'] * 100
        topic['rank'] = rank
        rank += 1
        results.append(topic)

    return results


def return_embedding(s: Search, vector_field: str) -> dict:
    # print(f'begin return_embedding for s.to_dict()={s.to_dict()}')

    s = s.execute()

    for hit in s.hits.hits:
        if vector_field in hit._source:
            return hit._source[vector_field]._l_


def return_all_embeddings(s: Search, vector_field: str) -> dict:
    # print(f'begin return_all_embeddings for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}

    for hit in s.hits.hits:
        if vector_field in hit._source:
            results[hit._id] = hit._source[vector_field]._l_

    return results
