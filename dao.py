from app.models import TranscriptSource, Episode, Scene, SceneEvent


async def upsert_raw_episode(raw_episode: TranscriptSource) -> TranscriptSource:
    try:
        print(f'Looking up RawEpisode matching show_key={raw_episode.show_key} external_key={raw_episode.external_key}')
        fetched_re = await TranscriptSource.get(show_key=raw_episode.show_key, external_key=raw_episode.external_key)
        print(f'Previous RawEpisode matching show_key={raw_episode.show_key} external_key={raw_episode.external_key} found, upserting')
        fetched_re.transcript_type = raw_episode.transcript_type
        fetched_re.transcript_url = raw_episode.transcript_url
        await fetched_re.save()
        return fetched_re
        
    except Exception as e:
        print(f'No previous stored RawEpisode matching show_key={raw_episode.show_key} external_key={raw_episode.external_key} found, inserting:', e)
        await raw_episode.save()
        return raw_episode


async def fetch_raw_episode(show_key: str, episode_key: str) -> TranscriptSource|Exception:
    try:
        print(f'Looking up RawEpisode matching show_key={show_key} external_key={episode_key}')
        fetched_re = await TranscriptSource.get(show_key=show_key, external_key=episode_key)
        return fetched_re
    except Exception as e:
        print(f'No RawEpisode found matching show_key={show_key} external_key={episode_key}:', e)
        raise e


async def fetch_episode(show_key: str, episode_key: str) -> Episode|Exception:
    try:
        print(f'Looking up Episode matching show={show_key} external_key={episode_key}')
        fetched_episode = await Episode.get(show_key=show_key, external_key=episode_key)
        return fetched_episode
    except Exception as e:
        print(f'No Episode found matching show_key={show_key} external_key={episode_key}:', e)
        raise e
    

async def delete_episode(episode: Episode) -> None|Exception:
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


async def insert_episode(episode: Episode) -> None|Exception:
    print(f'Begin insert_episode episode={episode}')
    try:
        await Episode.save(episode)
    except Exception as e:
        print(f'Failure during insert_episode to insert episode={episode}:', e)
        raise e
    print(f'Completed insert_episode episode={episode}')
    # return episode


async def insert_transcript(episode: Episode, scenes: list[Scene], scenes_to_events: dict[int, SceneEvent]) -> None|Exception:
    print(f'Begin insert_episode episode={episode} len(scenes)={len(scenes)} len(scenes_to_events)={len(scenes_to_events)}')
    for scene in scenes:
        scene.episode = episode
        try:
            await Scene.save(scene)
        except Exception as e:
            print(f'Failure during insert_transcript to insert scene={scene} for episode={episode}:', e)
            # raise e
        if scene.sequence_in_episode not in scenes_to_events:
            print(f'Failure during insert_transcript to insert events for scene={scene} episode={episode}: no events matching scene.sequence_in_episode={scene.sequence_in_episode} in scenes_to_events')
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
                print(f'Failure during insert_transcript to insert scene_event={event} in scene={scene} for episode={episode}:', e)
                # raise e
    print(f'Completed insert_transcript episode={episode}')


async def upsert_episode(episode: Episode) -> Episode:
    print(f'Begin upsert_episode episode={episode}')

    fetched_episode = None
    try:
        fetched_episode = await fetch_episode(episode.show_key, episode.external_key)
        # fetched_episode = Episode.filter(show_key=episode.show_key, external_key=episode.external_key)
    except Exception:
        pass
    
    if fetched_episode:
        # await fetched_episode.update(episode.update_from_dict())
        # https://github.com/tortoise/tortoise-orm/issues/424
        # fields = list(fetched_episode.__dict__.keys())

        # unwanted = ['_partial', '_saved_in_db', '_custom_generated_pk', 'id']
        # cleaned = [f for f in fields if f not in unwanted]
        # fetched_episode.update_from_dict(**episode.dict())
        fields = ['season', 'sequence_in_season', 'title', 'air_date', 'duration']
        # for field in episode.__class__._meta.fields:
        for field in fields:
            setattr(fetched_episode, field, getattr(episode, field))
        await fetched_episode.save()
        # await fetched_episode.save(update_fields=fields)
        # await fetched_episode.update(**episode.dict(), update_fields=cleaned)
        return fetched_episode
    else:
        try:
            # fetched_episode = await insert_episode(episode)
            await insert_episode(episode)
            return episode
        except Exception as e:
            print(f'Failure during upsert_episode to insert episode={episode}:', e)
            raise e
        
    # return fetched_episode
