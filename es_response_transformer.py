from elasticsearch_dsl import Search
from operator import itemgetter


async def return_episodes_by_title(s: Search) -> list:
    print(f'begin return_episodes_by_title for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = []

    for hit in s.hits.hits:
        episode = hit._source
        episode['score'] = hit._score
        if 'highlight' in hit:
            episode['title'] = hit['highlight']['title'][0]
        results.append(episode._d_)

    return results


async def return_scenes(s: Search) -> (list, int):
    print(f'begin return_scenes for s.to_dict()={s.to_dict()}')

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


async def return_scene_events(s: Search, location: str = None) -> (list, int, int):
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
        orig_scenes = episode.scenes

        scene_offset_to_scene = {}

        # highlight and map inner_hit scene_events to scenes using scene_offset
        for scene_event_hit in hit.inner_hits['scenes.scene_events'].hits.hits:
            scene_offset = scene_event_hit._nested.offset
            # NOTE hack to filter on location after es payload returned, until I find a way to cross-reference nested documents
            # or punt and enable `include_in_root`
            if location and orig_scenes[scene_offset]['location'] != location:
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
            scene_event_count += 1

        # NOTE follow-up to location-filter hack above, if all scenes have been filtered then skip episode 
        if len(scene_offset_to_scene) == 0:
            # print(f'dropping episode_key={episode["episode_key"]}, no scenes remain after location filtering')
            continue

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

    return results, scene_count, scene_event_count


async def return_episodes(s: Search) -> (list, int, int):
    print(f'begin return_episodes for s.to_dict()={s.to_dict()}')

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
        for scene_offset, scene in scene_offset_to_scene.items():
            episode.scenes.append(scene._d_)
            episode['agg_score'] += scene['agg_score']
            episode['high_child_score'] = max(scene['agg_score'], episode['high_child_score'])
            scene_count += 1
        results.append(episode._d_)

    # sort results before returning
    results = sorted(results, key=itemgetter('agg_score'), reverse=True)

    return results, scene_count, scene_event_count


async def return_scenes_by_location(s: Search, speaker: str = None) -> list:
    print(f'begin return_scenes_by_location for s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}

    if speaker:
        for item in s.aggregations.scene_events.speaker_match.scenes.by_location.buckets:
            results[item.key] = item.doc_count
    else:
        for item in s.aggregations.scenes.by_location.buckets:
            results[item.key] = item.doc_count

    return results


async def return_scenes_by_speaker(s: Search, location: str = None, other_speaker: str = None) -> list:
    print(f'begin return_scenes_by_speaker for location={location} other_speaker={other_speaker} s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}

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


async def return_scene_events_by_speaker(s: Search, dialog: str = None) -> list:
    print(f'begin return_scene_events_by_speaker for dialog={dialog} s.to_dict()={s.to_dict()}')

    s = s.execute()

    results = {}

    if dialog:
        for item in s.aggregations.scene_events.dialog_match.by_speaker.buckets:
            results[item.key] = item.doc_count
    else:
        for item in s.aggregations.scene_events.by_speaker.buckets:
            results[item.key] = item.doc_count

    return results