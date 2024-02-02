from bs4 import BeautifulSoup
from fastapi import APIRouter
import os
import requests
import shutil

import app.database.dao as dao
import app.etl.transcript_extractor as te
import app.etl.transcript_listing_extractor as tle
import app.pydantic_models as pymod
from app.show_metadata import ShowKey, show_metadata, WIKIPEDIA_DOMAIN
import app.utils as utils


etl_app = APIRouter()


@etl_app.get("/etl/copy_episode_listing/{show_key}", tags=['ETL'])
async def copy_episode_listing(show_key: ShowKey):
    '''
    Copies html of external episode listing page (as configured in `show_metadata`) to `source/episode_listings/` 
    '''
    dir_path, backup_dir_path = utils.get_or_make_source_dirs('episode_listings')
    # copy html from source file as configured in show metadata
    episode_listing_html = requests.get(WIKIPEDIA_DOMAIN + show_metadata[show_key]['wikipedia_label'])
    file_path = f'{dir_path}/{show_key.value}.html'
    # back up file if it already exists
    backup_file_path = None
    if os.path.isfile(file_path):
        backup_file_path = f'{backup_dir_path}/{show_key.value}.html'
        shutil.copyfile(file_path, backup_file_path)
    # write to file
    with open(file_path, "w") as f:
        f.write(episode_listing_html.text)
    return {'show_key': show_key.value, 'file_path': file_path, 'backup_file_path': backup_file_path, 'episode_listing_html': episode_listing_html.text}


@etl_app.get("/etl/copy_transcript_sources/{show_key}", tags=['ETL'])
async def copy_transcript_sources(show_key: ShowKey):
    '''
    Copies html of external transcript url listing page (as configured in `show_metadata`) to `source/transcript_sources/`
    '''
    dir_path, backup_dir_path = utils.get_or_make_source_dirs('transcript_sources')
    # copy html from source file as configured in show metadata
    show_transcripts_domain = show_metadata[show_key]['show_transcripts_domain']
    listing_url = show_metadata[show_key]['listing_url']
    transcript_source_html = requests.get(show_transcripts_domain + listing_url)
    file_path = f'{dir_path}/{show_key.value}.html'
    # back up file if it already exists
    backup_file_path = None
    if os.path.isfile(file_path):
        backup_file_path = f'{backup_dir_path}/{show_key.value}.html'
        shutil.copyfile(file_path, backup_file_path)
    # write to file
    with open(file_path, "w") as f:
        f.write(transcript_source_html.text)
    return {'show_key': show_key.value, 'file_path': file_path, 'backup_file_path': backup_file_path, 'transcript_source_html': transcript_source_html.text}
    

@etl_app.get("/etl/copy_transcript_from_source/{show_key}/{episode_key}", tags=['ETL'])
async def copy_transcript_from_source(show_key: ShowKey, episode_key: str):
    '''
    Copies html of external episode page (fetched from `TranscriptSource`) to `source/episodes/`
    '''
    dir_path, backup_dir_path = utils.get_or_make_source_dirs('episodes', show_key.value)
    # fetch episode and transcript_source(s), throw errors if not found
    episode = None
    try:
        episode = await dao.fetch_episode(show_key.value, episode_key, fetch_related=['transcript_sources'])
        if not episode.transcript_sources:
            return {"Error": f"No Transcript found for episode having show_key={show_key} external_key={episode_key}. You may need to run /load_transcript_sources first."}
    except Exception as e:
        return {"Error": f"Failure to fetch Episode having show_key={show_key} external_key={episode_key} (have you run /load_episode_listing?): {e}"}
    if not episode:
        return {"Error": f"No Episode found having show_key={show_key} external_key={episode_key}. You may need to run /load_episode_listing first."}
    
    # TODO data model permits multiple transcript_sources per episode, for now just choose first one
    # TODO ultimately the /source/episodes file structure will need to reflect the transcript_source layer
    transcript_source = episode.transcript_sources[0]
    transcript_html = requests.get(transcript_source.transcript_url)

    file_path = f'{dir_path}/{episode_key}.html'
    # back up file if it already exists
    backup_file_path = None
    if os.path.isfile(file_path):
        backup_file_path = f'{backup_dir_path}/{episode_key}.html'
        shutil.copyfile(file_path, backup_file_path)
    # write to file
    with open(file_path, "w") as f:
        f.write(transcript_html.text)
    
    return {'show_key': show_key.value, 'episode_key': episode_key, 'file_path': file_path, 'backup_file_path': backup_file_path, 'transcript_html': transcript_html.text}


@etl_app.get("/etl/copy_all_transcripts_from_source/{show_key}", tags=['ETL'])
async def copy_all_transcripts_from_source(show_key: ShowKey):
    '''
    Bulk copies html of external episode pages (fetched from `TranscriptSource` entities) to `source/episodes/`. Bulk equivalent of `/etl/copy_transcript_from_source`.
    '''
    dir_path, backup_dir_path = utils.get_or_make_source_dirs('episodes', show_key.value)
    # fetch episodes from db
    episodes = []
    try:
        episodes = await dao.fetch_episodes(show_key.value)
    except Exception as e:
        return {"Error": f"Failure to fetch Episodes having show_key={show_key}: {e}"}
    if not episodes:
        return {"Error": f"No Episodes found having show_key={show_key}. You may need to run /load_episode_listing first."}

    # load transcripts for all episodes into db
    attempts = 0
    no_transcript_episode_keys = []
    successful_episode_keys = []
    failed_episode_keys = []
    for episode in episodes:
        await episode.fetch_related('transcript_sources')
        if not episode.transcript_sources:
            print(f"No Transcript found for episode having show_key={show_key.value} external_key={episode.external_key}. You may need to run /load_transcript_sources first.")
            no_transcript_episode_keys.append(episode.external_key)
            continue
        # TODO data model permits multiple transcript_sources per episode, for now just choose first one
        # TODO ultimately the /source/episodes file structure will need to reflect the transcript_source layer
        transcript_source = episode.transcript_sources[0]
        attempts += 1
        try:
            transcript_html = requests.get(transcript_source.transcript_url)
            file_path = f'{dir_path}/{episode.external_key}.html'
            # back up file if it already exists
            if os.path.isfile(file_path):
                backup_file_path = f'{backup_dir_path}/{episode.external_key}.html'
                shutil.copyfile(file_path, backup_file_path)
            # write to file
            with open(file_path, "w") as f:
                f.write(transcript_html.text)
                successful_episode_keys.append(episode.external_key)
        except Exception as e:
            failed_episode_keys.append(episode.external_key)
            print(f"Failure to copy episode {show_key}:{episode.external_key} from url={transcript_source.transcript_url}: {e}")

    return {
        "no transcripts": len(no_transcript_episode_keys),
        "no transcripts episode keys": no_transcript_episode_keys,
        "transcript copy attempts": attempts, 
        "successful": len(successful_episode_keys),
        "successful episode keys": successful_episode_keys, 
        "failed": len(failed_episode_keys),
        "failed episode keys": failed_episode_keys, 
    }
   

################### WRITE EXTERNALLY SOURCED EPISODE LISTING METADATA TO DB ###################

@etl_app.get("/etl/load_episode_listing/{show_key}", tags=['ETL'])
async def load_episode_listing(show_key: ShowKey, write_to_db: bool = False):
    '''
    Load raw episode listing html from `source/episode_listings/` into transcript_db. Initializes `Episode` db entities in Postgres.
    '''
    file_path = f'source/episode_listings/{show_key.value}.html'
    if not os.path.isfile(file_path):
        return {'Error': f'Unable to load episode metadata for {show_key.value}, no source html at file_path={file_path} (have you run /copy_episode_listing?)'}

    episode_listing_soup = BeautifulSoup(open(file_path).read(), 'html5lib')
    episodes = tle.parse_episode_listing_soup(show_key, episode_listing_soup)

    # NOTE used to create test data, haven't decided how to retain these steps
    # import json
    # json_episodes = []
    # for episode in episodes:
    #     episode_pyd = await pymod.EpisodePydanticExcluding.from_tortoise_orm(episode)
    #     episode_json = episode_pyd.model_dump_json()
    #     json_episodes.append(episode_json)

    # with open(f"test_data/responses/parse_episode_listing_soup_{show_key}.json", "w") as file:
    #     json.dump(json_episodes, file, indent=4)

    if write_to_db:
        stored_episodes = []
        for episode in episodes:
            stored_episode = await dao.upsert_episode(episode)
            stored_episodes.append(stored_episode)
        return {'episode_count': len(stored_episodes), 'write_to_db': write_to_db, 'episode_listing': stored_episodes}
    else:
        episodes_excl = []
        for episode in episodes:
            episode_excl = await pymod.EpisodePydanticExcluding.from_tortoise_orm(episode)
            episodes_excl.append(episode_excl)
        return {'episode_count': len(episodes_excl), 'write_to_db': write_to_db, 'episodes': episodes_excl}


@etl_app.get("/etl/load_transcript_sources/{show_key}", tags=['ETL'])
async def load_transcript_sources(show_key: ShowKey, write_to_db: bool = False):
    '''
    Load raw transcript source html from `source/transcript_sources/` into transcript_db. Initializes `TranscriptSource` db entities.
    '''
    file_path = f'source/transcript_sources/{show_key.value}.html'
    if not os.path.isfile(file_path):
        return {'Error': f'Unable to load transcript sources for {show_key.value}, no source html at file_path={file_path} (have you run /copy_transcript_sources?)'}

    transcript_sources_soup = BeautifulSoup(open(file_path).read(), 'html5lib')
    episode_transcripts_by_type = tle.parse_transcript_url_listing_soup(show_key, transcript_sources_soup)
    transcript_sources = await tle.match_episodes_to_transcript_urls(show_key, episode_transcripts_by_type)
    if write_to_db:
        stored_tx_sources = []
        for tx_source in transcript_sources:
            stored_tx_source = await dao.upsert_transcript_source(tx_source)
            stored_tx_sources.append(stored_tx_source)
        return {'transcript_sources_count': len(stored_tx_sources), 'transcript_sources': stored_tx_sources}
    else:
        return {'transcript_sources_count': len(transcript_sources), 'transcript_sources': transcript_sources}


@etl_app.get("/etl/load_transcript/{show_key}/{episode_key}", tags=['ETL'])
async def load_transcript(show_key: ShowKey, episode_key: str, write_to_db: bool = False):
    '''
    Parse and load transcript html from `source/episodes/` to transcript_db. Generates `Scene` and `SceneEvent` db entities.
    '''
    episode = None
    # fetch episode and transcript_source(s), throw errors if not found
    try:
        episode = await dao.fetch_episode(show_key.value, episode_key, fetch_related=['transcript_sources'])
        if not episode.transcript_sources:
            return {"Error": f"No Transcript found for episode having show_key={show_key} external_key={episode_key}. You may need to run /load_transcript_sources first."}
    except Exception as e:
        return {"Error": f"Failure to fetch Episode having show_key={show_key} external_key={episode_key} (have you run /load_episode_listing?): {e}"}
    if not episode:
        return {"Error": f"No Episode found having show_key={show_key} external_key={episode_key}. You may need to run /load_episode_listing first."}
    
    # TODO data model permits multiple transcript_sources per episode, for now just choose first one
    # TODO ultimately the /source/episodes file structure will need to reflect the transcript_source layer
    transcript_source = episode.transcript_sources[0]
    file_path = f'source_override/episodes/{show_key.value}/{episode_key}.html'
    if not os.path.isfile(file_path):
        file_path = f'source/episodes/{show_key.value}/{episode_key}.html'
        if not os.path.isfile(file_path):
            return {'Error': f'Unable to load transcript for {show_key}:{episode_key}, no source html at file_path={file_path} (have you run /copy_transcript_from_source?)'}
    transcript_soup = BeautifulSoup(open(file_path).read(), 'html5lib')
    
    # transform raw transcript into persistable scene and scene_event data
    scenes, scenes_to_events = te.parse_episode_transcript_soup(episode, transcript_source.transcript_type, transcript_soup)
    print(f'len(scenes)={len(scenes)}')
    
    if write_to_db:
        await dao.insert_transcript(episode, scenes=scenes, scenes_to_events=scenes_to_events)
        episode_pyd = await pymod.EpisodePydantic.from_tortoise_orm(episode)
        return {'show': show_metadata[show_key], 'episode': episode_pyd}
    else:
        # this response should be identical to the persisted version above
        episode_excl = await pymod.EpisodePydanticExcluding.from_tortoise_orm(episode)
        episode_excl.scenes = []  # we don't want to include any scenes fetched from previously persisted episode
        for scene in scenes:
            scene_excl = await pymod.ScenePydanticExcluding.from_tortoise_orm(scene)
            episode_excl.scenes.append(scene_excl)
            if scene_excl.sequence_in_episode in scenes_to_events:
                for i in range(len(scenes_to_events[scene_excl.sequence_in_episode])):
                    scene_event = scenes_to_events[scene_excl.sequence_in_episode][i]
                    scene_event.sequence_in_scene = i+1
                    # TODO I am STUMPED as to why scene_event.id must be set while scene.id and episode.id do not
                    scene_event.id = i+1
                    scene_event_excl = await pymod.SceneEventPydanticExcluding.from_tortoise_orm(scene_event)
                    scene_excl.events.append(scene_event_excl)
        return {'show': show_metadata[show_key], 'episode': episode_excl}


@etl_app.get("/etl/load_all_transcripts/{show_key}", tags=['ETL'])
async def load_all_transcripts(show_key: ShowKey, overwrite_all: bool = False):
    '''
    Parse and load transcript html from `source/episodes/` to transcript_db. Bulk equivalent of `/etl/load_transcript/`.
    '''
    episodes = []
    try:
        episodes = await dao.fetch_episodes(show_key.value)
    except Exception as e:
        return {"Error": f"Failure to fetch Episodes having show_key={show_key}: {e}"}
    if not episodes:
        return {"Error": f"No Episodes found having show_key={show_key}. You may need to run /load_episode_listing first."}
    if not overwrite_all:
        return {"No-op": f"/load_transcripts was invoked on {len(episodes)} episodes, but `overwrite_all` flag was not set to True so no action was taken"}

    # fetch and insert transcripts for all episodes
    attempts = 0
    no_transcript_episode_keys = []
    successful_episode_keys = []
    failed_episode_keys = []
    for episode in episodes:
        await episode.fetch_related('transcript_sources')
        if not episode.transcript_sources:
            print(f"No Transcript found for episode having show_key={show_key.value} external_key={episode.external_key}. You may need to run /load_transcript_sources first.")
            no_transcript_episode_keys.append(episode.external_key)
            continue

        # TODO data model permits multiple transcript_sources per episode, for now just choose first one
        # TODO ultimately the /source/episodes file structure will need to reflect the transcript_source layer
        transcript_source = episode.transcript_sources[0]
        file_path = f'source_override/episodes/{show_key.value}/{episode.external_key}.html'
        if not os.path.isfile(file_path):
            file_path = f'source/episodes/{show_key.value}/{episode.external_key}.html'
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
            
    return {
        "no transcripts": len(no_transcript_episode_keys),
        "no transcripts episode keys": no_transcript_episode_keys,
        "transcript load attempts": attempts, 
        "successful": len(successful_episode_keys),
        "successful episode keys": successful_episode_keys, 
        "failed": len(failed_episode_keys),
        "failed episode keys": failed_episode_keys, 
    }
