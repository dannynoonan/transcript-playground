from datetime import datetime, timezone
import subprocess

from app.models import TranscriptSource, Episode, Scene, SceneEvent


async def backup_db() -> tuple[bytes, bytes]:
    bashCommand = "bash backup_db.sh"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    return process.communicate()


async def fetch_episodes(show_key: str) -> list[Episode]|Exception:
    print(f'Fetching all episodes matching show={show_key}')
    try:
        fetched_episodes = await Episode.filter(show_key=show_key)
        return fetched_episodes
    except Exception as e:
        print(f'Failure to fetch episodes matching show_key={show_key}:', e)
        raise e
    

async def fetch_episode(show_key: str, episode_key: str, fetch_related: list = []) -> Episode|Exception:
    print(f'Looking up Episode matching show={show_key} external_key={episode_key}')
    try:
        fetched_episode = await Episode.get(show_key=show_key, external_key=episode_key)
        for rel in fetch_related:
            if rel == 'events':
                for scene in fetched_episode.scenes:
                    await scene.fetch_related('events')
            else:
                await fetched_episode.fetch_related(rel)
        return fetched_episode
    except Exception as e:
        print(f'No Episode found matching show_key={show_key} external_key={episode_key}:', e)
        raise e


async def insert_episode(episode: Episode) -> None|Exception:
    print(f'Begin insert_episode episode={episode}')
    try:
        # set loaded_ts
        episode.loaded_ts = datetime.now(timezone.utc)
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
        # set loaded_ts
        fetched_episode.loaded_ts = datetime.now(timezone.utc)
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
    print(f'Looking up TranscriptSource matching show={show_key} episode_key={episode_key} transcript_type={transcript_type}')
    try:
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
    print(f'Begin insert_episode for episode={episode} len(scenes)={len(scenes)}')

    # delete any scene data previously mapped to episode
    await episode.fetch_related('scenes')
    if episode.scenes:
        print(f'Previously mapped scene data found for episode={episode}, deleting before inserting new scene data...')
        for old_scene in episode.scenes:
            await old_scene.delete()

    # insert scene data for episode
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
    
    # set transcript_loaded_ts
    episode.transcript_loaded_ts = datetime.now(timezone.utc)
    await episode.save()

    print(f'Completed insert_transcript for episode={episode}')
