import argparse
import asyncio
import os
import sys
from tortoise import Tortoise
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from app.config import DATABASE_URL
from app.database import dao
import app.es.es_ingest_transformer as esit
import app.es.es_query_builder as esqb


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    print(f'Begin index_episodes script for show_key={show_key}')

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
        return {"Error": f"No Episodes found having show_key={show_key}. You may need to run /load_episode_listing first."}
    # if not overwrite_all:
    #     return {"No-op": f"/index_all_episodes was invoked on {len(episodes)} episodes, but `overwrite_all` flag was not set to True so no action was taken"}
    print(f'Fetched {len(episodes)} db episodes for show_key={show_key}. Begin upserting to es transcripts index.')
    
    # fetch and insert transcripts for all episodes
    attempts = 0
    successful_episode_keys = []
    failed_episode_keys = []
    for episode in episodes:
        attempts += 1
        print(f'Begin indexing episode {episode}')

        # fetch nested scene and scene_event data
        await episode.fetch_related('scenes')
        for scene in episode.scenes:
            await scene.fetch_related('events')

        # transform to es-writable object and write to es
        try:
            es_episode = esit.to_es_episode(episode)
            esqb.save_es_episode(es_episode)
            successful_episode_keys.append(episode.external_key)
        except Exception as e:
            failed_episode_keys.append(episode.external_key)
            print(f"Failure to transform Episode {show_key}_{episode.external_key} to es-writable version or write it to es: {e}")

    # TODO populate focal speakers / locations here

    report = {
        "episode_indexing_attempts": attempts, 
        "successful": len(successful_episode_keys),
        "successful_episode_keys": successful_episode_keys, 
        "failed": len(failed_episode_keys),
        "failed_episode_keys": failed_episode_keys, 
    }
    print(report)
    return report


# if __name__ == '__main__':
#     main()

if __name__ == "__main__":
    asyncio.run(main())
