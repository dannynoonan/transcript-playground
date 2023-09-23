from app.models import RawEpisodeMap


async def upsert_raw_episode(raw_episode: RawEpisodeMap) -> None:
    try:
        print(f'Looking up RawEpisodeMap matching show_key={raw_episode.show_key} external_key={raw_episode.external_key}')
        # fetched_re = await RawEpisodeMap.filter(show_key=raw_episode.show_key, external_key=raw_episode.external_key).first()
        fetched_re = await RawEpisodeMap.get(show_key=raw_episode.show_key, external_key=raw_episode.external_key)
        print(f'Previous RawEpisodeMap matching show_key={raw_episode.show_key} external_key={raw_episode.external_key} found, upserting')
        fetched_re.transcript_type = raw_episode.transcript_type
        fetched_re.transcript_url = raw_episode.transcript_url
        await fetched_re.save()
        # raw_episode = fetched_re
        
    except Exception as e:
        print(f'No previous stored RawEpisodeMap matching show_key={raw_episode.show_key} external_key={raw_episode.external_key} found, inserting', e)
        print(f'raw_episode={raw_episode}')
        await raw_episode.save()
    
    # await raw_episode.save() 