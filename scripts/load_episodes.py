import argparse
import asyncio
from bs4 import BeautifulSoup
import os
import sys
from tortoise import Tortoise
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from app.config import DATABASE_URL
from app.database import dao
import app.es.es_ingest_transformer as esit
import app.es.es_query_builder as esqb
import app.etl.transcript_extractor as te


async def main():
    '''
    Bulk run of `/etl/load_transcript` for all episodes of a given show. Parse and load transcript html from `source/episodes/` to transcript_db. 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    print(f'Begin load_episodes script for show_key={show_key}')

    await Tortoise.init(
        db_url=DATABASE_URL,
        modules={'models': ['app.models']}
    )

    episodes = []
    try:
        episodes = await dao.fetch_episodes(show_key)
    except Exception as e:
        return {"Error": f"Failure to fetch Episodes having show_key={show_key}: {e}"}
    if not episodes:
        return {"Error": f"No Episodes found having show_key={show_key}. You may need to run /etl/load_episode_listing first."}
    print(f'Fetched {len(episodes)} db episodes for show_key={show_key}. Begin loading transcript data from file sources into postgres.')
    
    # fetch and insert transcripts for all episodes
    attempts = 0
    no_transcript_episode_keys = []
    successful_episode_keys = []
    failed_episode_keys = []
    for episode in episodes:
        print(f'Begin loading episode {episode}')
        await episode.fetch_related('transcript_sources')
        if not episode.transcript_sources:
            print(f"No Transcript found for episode having show_key={show_key} external_key={episode.external_key}. You may need to run /etl/load_transcript_sources first.")
            no_transcript_episode_keys.append(episode.external_key)
            continue

        # TODO data model permits multiple transcript_sources per episode, for now just choose first one
        # TODO ultimately the /source/episodes file structure will need to reflect the transcript_source layer
        transcript_source = episode.transcript_sources[0]
        file_path = f'source_override/episodes/{show_key}/{episode.external_key}.html'
        if not os.path.isfile(file_path):
            file_path = f'source/episodes/{show_key}/{episode.external_key}.html'
            if not os.path.isfile(file_path):
                failed_episode_keys.append(episode.external_key)
                continue
        
        # fetch and transform raw transcript into persistable scene and scene_event data
        transcript_soup = BeautifulSoup(open(file_path).read(), 'html5lib')
        scenes, scenes_to_events = te.parse_episode_transcript_soup(episode, transcript_source.transcript_type, transcript_soup)
        attempts += 1
        try:
            await dao.insert_transcript(episode, scenes=scenes, scenes_to_events=scenes_to_events)
            successful_episode_keys.append(episode.external_key)
        except Exception as e:
            failed_episode_keys.append(episode.external_key)
            print(f"Failure to insert Episode having show_key={show_key} external_key={episode.external_key}: {e}")
            
    print(f'Successfully loaded transcripts for {len(successful_episode_keys)} of {len(episodes)} fetched episodes into transcript_db.')

    report = {
        "no_transcripts": len(no_transcript_episode_keys),
        "no_transcripts_episode_keys": no_transcript_episode_keys,
        "transcript_load_attempts": attempts, 
        "successful": len(successful_episode_keys),
        "successful_episode_keys": successful_episode_keys, 
        "failed": len(failed_episode_keys),
        "failed_episode_keys": failed_episode_keys, 
    }
    print(report)
    return report


if __name__ == "__main__":
    asyncio.run(main())
