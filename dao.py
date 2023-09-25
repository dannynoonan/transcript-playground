from app.models import Show, RawEpisode, Episode, Scene, SceneEvent


async def upsert_raw_episode(raw_episode: RawEpisode) -> None:
    try:
        print(f'Looking up RawEpisode matching show_key={raw_episode.show_key} external_key={raw_episode.external_key}')
        # fetched_re = await RawEpisode.filter(show_key=raw_episode.show_key, external_key=raw_episode.external_key).first()
        fetched_re = await RawEpisode.get(show_key=raw_episode.show_key, external_key=raw_episode.external_key)
        print(f'Previous RawEpisode matching show_key={raw_episode.show_key} external_key={raw_episode.external_key} found, upserting')
        fetched_re.transcript_type = raw_episode.transcript_type
        fetched_re.transcript_url = raw_episode.transcript_url
        await fetched_re.save()
        # raw_episode = fetched_re
        
    except Exception as e:
        print(f'No previous stored RawEpisode matching show_key={raw_episode.show_key} external_key={raw_episode.external_key} found, inserting:', e)
        print(f'raw_episode={raw_episode}')
        await raw_episode.save()
    
    # await raw_episode.save() 


async def fetch_raw_episode(show_key: str, episode_key: str) -> RawEpisode|Exception:
    try:
        print(f'Looking up RawEpisode matching show_key={show_key} external_key={episode_key}')
        # fetched_re = await RawEpisode.filter(show_key=show_key, external_key=episode_key).first()
        fetched_re = await RawEpisode.get(show_key=show_key, external_key=episode_key)
        return fetched_re
    except Exception as e:
        print(f'No RawEpisode found matching show_key={show_key} external_key={episode_key}:', e)
        raise e


async def fetch_episode(show_key: str, episode_key: str) -> Episode|Exception:
    try:
        fetched_show = await fetch_show(show_key)
        print(f'Looking up Episode matching show={fetched_show} external_key={episode_key}')
        fetched_episode = await Episode.get(show=fetched_show, external_key=episode_key)
        return fetched_episode
    except Exception as e:
        print(f'No Episode found matching show_key={show_key} external_key={episode_key}:', e)
        raise e
    

async def delete_episode(episode: Episode):
# async def delete_episode(show_key: str, episode_key: str):
    print(f'Begin delete_episode={episode}')
    # print(f'Begin delete_episode show_key={show_key} episode_key={episode_key}')
    try:
        # deleted_episode = await episode.delete()
        await episode.delete()
    except Exception as e:
        print(f'Failure 1 to delete episode={episode}:', e)
        raise e
    # if not deleted_episode:
    #     print(f'Failure 2 to delete episode={episode}')
    #     raise(f'Failure 3 to delete episode={episode}')


async def insert_episode(episode: Episode, scenes: list[Scene], scenes_to_events: dict[int, SceneEvent]):
    print(f'Begin insert_episode episode={episode} len(scenes)={len(scenes)} len(scenes_to_events)={len(scenes_to_events)}')
    try:
        await Episode.save(episode)
    except Exception as e:
        print(f'Failure during insert_episode to insert episode={episode}:', e)
        raise e
    for scene in scenes:
        scene.episode = episode
        # print(f'++++++++ Begin insert for scene={scene} of type={type(scene)}')
        try:
            await Scene.save(scene)
        except Exception as e:
            print(f'Failure during insert_episode to insert scene={scene} for episode={episode}:', e)
            # raise e
        # print(f'++++++++ Finish inserting scene={scene} which now has id={scene.id}')
        if scene.sequence_in_episode not in scenes_to_events:
            print(f'Failure during insert_episode to insert events for scene={scene} episode={episode}: no events matching scene.sequence_in_episode={scene.sequence_in_episode} in scenes_to_events')
            continue
        scene_events = scenes_to_events[scene.sequence_in_episode]
        event_i = 1
        for event in scene_events:
            try:
                event.scene = scene
                event.sequence_in_scene = event_i
                await SceneEvent.save(event)
                event_i += 1
            except Exception as e:
                print(f'Failure during insert_episode to insert scene_event={event} in scene={scene} for episode={episode}:', e)
                # raise e
    print(f'Completed insert_episode episode={episode}')


async def upsert_episode(episode: Episode, scenes: list[Scene], scenes_to_events: dict):
    print(f'Begin upsert_episode episode={episode} len(scenes)={len(scenes)} len(scenes_to_events)={len(scenes_to_events)}')

    try:
        fetched_episode = await fetch_episode(episode.show.key, episode.external_key)
        if fetched_episode:
            try:
                # await delete_episode(episode.show.key, episode.external_key)
                await delete_episode(fetched_episode)
            except Exception as e:
                print(f'Failure during upsert_episode to delete episode={fetched_episode}:', e)
                raise e
    except Exception:
        pass

    # print(f'@@@@@@ Midway thru upsert_episode')

    try:
        await insert_episode(episode, scenes, scenes_to_events)
    except Exception as e:
        print(f'Failure during upsert_episode to insert episode={fetched_episode}:', e)
        raise e


async def fetch_show(show_key: str) -> Show|Exception:
    print(f'Begin fetch_show for show_key={show_key}')
    try:
        # fetched_show = await Show.filter(show_key=show_key).first()
        fetched_show = await Show.get(key=show_key)
        return fetched_show
    except Exception as e:
        print(f'No Show found matching show_key={show_key}:', e)
        raise e
