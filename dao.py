from app.models import TranscriptSource, Episode, Scene, SceneEvent


async def fetch_episodes(show_key: str) -> list[Episode]|Exception:
    try:
        print(f'Fetching all episodes matching show={show_key}')
        fetched_episodes = await Episode.filter(show_key=show_key)
        return fetched_episodes
    except Exception as e:
        print(f'Failure to fetch episodes matching show_key={show_key}:', e)
        raise e
    

async def fetch_episode(show_key: str, episode_key: str) -> Episode|Exception:
    try:
        print(f'Looking up Episode matching show={show_key} external_key={episode_key}')
        fetched_episode = await Episode.get(show_key=show_key, external_key=episode_key)
        return fetched_episode
    except Exception as e:
        print(f'No Episode found matching show_key={show_key} external_key={episode_key}:', e)
        raise e


async def insert_episode(episode: Episode) -> None|Exception:
    print(f'Begin insert_episode episode={episode}')
    try:
        await Episode.save(episode)
    except Exception as e:
        print(f'Failure during insert_episode to insert episode={episode}:', e)
        raise e
    print(f'Completed insert_episode episode={episode}')
    # return episode


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


async def delete_episode(episode: Episode) -> None|Exception:
    print(f'Begin delete_episode={episode}')
    try:
        await episode.delete()
    except Exception as e:
        print(f'Failure 1 to delete episode={episode}:', e)
        raise e


async def fetch_transcript_source(show_key: str, episode_key: str, transcript_type: str) -> TranscriptSource|Exception:
    try:
        print(f'Looking up TranscriptSource matching show={show_key} episode_key={episode_key} transcript_type={transcript_type}')
        fetched_transcript_source = await TranscriptSource.get(episode__show_key=show_key, episode__external_key=episode_key, transcript_type=transcript_type)
        return fetched_transcript_source
    except Exception as e:
        print(f'No TranscriptSource found matching show_key={show_key} external_key={episode_key} transcript_type={transcript_type}:', e)
        raise e


async def insert_transcript_source(transcript_source: TranscriptSource) -> None|Exception:
    print(f'Begin insert transcript_source={transcript_source}')
    try:
        await TranscriptSource.save(transcript_source)
    except Exception as e:
        print(f'Failure during insert_transcript_source to insert transcript_source={transcript_source}:', e)
        raise e
    print(f'Completed insert transcript_source={transcript_source}')


async def upsert_transcript_source(transcript_source: TranscriptSource) -> TranscriptSource:
    print(f'Begin upsert transcript_source={transcript_source}')

    fetched_tx_source = None
    try:
        fetched_tx_source = await fetch_transcript_source(transcript_source.episode.show_key, transcript_source.episode.external_key, transcript_source.transcript_type)
    except Exception:
        pass

    if fetched_tx_source:
        fields = ['transcript_url']
        for field in fields:
            setattr(fetched_tx_source, field, getattr(transcript_source, field))
        await fetched_tx_source.save()
        return fetched_tx_source
    else:
        try:
            await insert_transcript_source(transcript_source)
            return transcript_source
        except Exception as e:
            print(f'Failure during upsert_transcript_source to insert transcript_source={transcript_source}:', e)
            raise e





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
