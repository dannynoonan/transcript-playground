from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
# from fastapi.responses import JSONResponse
# import json
# from tortoise import HTTPException
from tortoise.contrib.fastapi import HTTPNotFoundError, register_tortoise
from tortoise.contrib.pydantic import pydantic_model_creator
from tortoise import Tortoise

from app.models import Job, TranscriptSource, Episode, Scene, SceneEvent
import dao
from database.connect import connect_to_database
from show_metadata import ShowKey, Status, show_metadata
from soup_brewer import get_episode_detail_listing_soup, get_transcript_url_listing_soup, get_transcript_soup
from transcript_extractor import parse_episode_transcript_soup
from transcript_listing_extractor import parse_episode_listing_soup, parse_transcript_url_listing_soup, match_episodes_to_transcript_urls

# https://levelup.gitconnected.com/handle-registration-in-fastapi-and-tortoise-orm-2dafc9325b7a


# DATABASE_URL=f"postgres://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}?sslmode=disable"
# DATABASE_URL="postgres://postgres@localhost:5432/transcript_db?sslmode=disable"
# DATABASE_URL="postgres://andyshirey@localhost:5432/transcript_db?sslmode=disable"
DATABASE_URL="postgres://andyshirey@localhost:5432/transcript_db"


transcript_playground_app = FastAPI()


##### BEGIN THESE DO THE SAME THING ######
async def init():
    # Here we create a SQLite DB using file "db.sqlite3"
    #  also specify the app name of "models"
    #  which contain models from "app.models"
    await Tortoise.init(
        db_url=DATABASE_URL,
        modules={'models': ['app.models']}
    )
    # Generate the schema
    await Tortoise.generate_schemas()

register_tortoise(
    transcript_playground_app,
    # db_url="sqlite://db.sqlite3",
    db_url=DATABASE_URL,
    modules={"models": ["app.models"]},
    generate_schemas=True,
    add_exception_handlers=True,
)
##### END THESE DO THE SAME THING ######

Tortoise.init_models(["app.models"], "models")


job_pydantic = pydantic_model_creator(Job)
job_pydantic_no_ids = pydantic_model_creator(Job, exclude_readonly=True)

Raw_Episode_Pydantic = pydantic_model_creator(TranscriptSource)
Episode_Pydantic = pydantic_model_creator(Episode)
Scene_Pydantic = pydantic_model_creator(Scene)
Scene_Event_Pydantic = pydantic_model_creator(SceneEvent)

Raw_Episode_Pydantic_Excluding = pydantic_model_creator(TranscriptSource, exclude=("id", "loaded_ts"))
Episode_Pydantic_Excluding = pydantic_model_creator(Episode, exclude=("id", "loaded_ts"))
Scene_Pydantic_Excluding = pydantic_model_creator(Scene, exclude=("id", "episode", "episode_id"))
Scene_Event_Pydantic_Excluding = pydantic_model_creator(SceneEvent, exclude=("id", "scene", "scene_id"))

# print(f'Episode_Pydantic.model_json_schema()={Episode_Pydantic.model_json_schema()}')


@transcript_playground_app.get("/")
def root():
    return {"message": "Hi there World"}


@transcript_playground_app.get("/db_connect")
async def db_connect():
    await connect_to_database()
    return {"DB connection": "Indeed"}


@transcript_playground_app.get("/show_meta/{show_key}")
async def fetch_show_meta(show_key: ShowKey):
    show_meta = show_metadata[show_key]
    return {show_key: show_meta}


@transcript_playground_app.get("/load_episode_listing/{show_key}")
async def load_episode_listing(show_key: ShowKey, write_to_db: bool = False):
    episode_detail_listing_soup = await get_episode_detail_listing_soup(show_key)
    episodes = await parse_episode_listing_soup(show_key, episode_detail_listing_soup)
    if write_to_db:
        stored_episodes = []
        for episode in episodes:
            stored_episode = await dao.upsert_episode(episode)
            stored_episodes.append(stored_episode)
        return {'episode_count': len(stored_episodes), 'episode_listing': stored_episodes}
    else:
        episodes_excl = []
        for episode in episodes:
            episode_excl = await Episode_Pydantic_Excluding.from_tortoise_orm(episode)
            episodes_excl.append(episode_excl)
        return {'episode_count': len(episodes_excl), 'episodes': episodes_excl}


@transcript_playground_app.get("/load_transcript_sources/{show_key}")
async def load_transcript_sources(show_key: ShowKey, write_to_db: bool = False):
    listing_soup = await get_transcript_url_listing_soup(show_key)
    episode_transcripts_by_type = await parse_transcript_url_listing_soup(show_key, listing_soup)
    transcript_sources = await match_episodes_to_transcript_urls(show_key, episode_transcripts_by_type)
    if write_to_db:
        stored_tx_sources = []
        for tx_source in transcript_sources:
            stored_tx_source = await dao.upsert_transcript_source(tx_source)
            stored_tx_sources.append(stored_tx_source)
        return {'transcript_sources_count': len(stored_tx_sources), 'transcript_sources_listing': stored_tx_sources}
    else:
        return {'transcript_sources_count': len(transcript_sources), 'transcript_sources': transcript_sources}


@transcript_playground_app.get("/load_transcript/{show_key}/{episode_key}")
async def load_transcript(show_key: ShowKey, episode_key: str, write_to_db: bool = False):
    try:
        raw_episode = await dao.fetch_raw_episode(show_key.value, episode_key)
    except Exception as e:
        return {"Error": f"Failure to fetch RawEpisode having show_key={show_key} external_key={episode_key}: {e}"}
    episode_soup = await get_transcript_soup(raw_episode)
    episode, scenes, scenes_to_events = await parse_episode_transcript_soup(show_key, episode_key, raw_episode.transcript_type, episode_soup)

    if write_to_db:
        await dao.upsert_episode(episode, scenes=scenes, scenes_to_events=scenes_to_events)
        episode_pyd = await Episode_Pydantic.from_tortoise_orm(episode)
        return {'show': show_metadata[show_key], 'raw_episode': raw_episode, 'episode': episode_pyd}

    else:
        # this response should be identical to the persisted version above
        # show_excluding = await Show_Pydantic_Excluding.from_tortoise_orm(episode.show)
        # print(f'^^^^^^^^ episode={episode}')
        # print(f'^^^^^^^^ vars(episode)={vars(episode)}')
        episode_excl = await Episode_Pydantic_Excluding.from_tortoise_orm(episode)
        # scenes_excluding = []
        for scene in scenes:
            # print(f'++++++++ scene={scene}')
            # print(f'++++++++ vars(scene)={vars(scene)}')
            # scene.episode = episode_excl
            scene_excl = await Scene_Pydantic_Excluding.from_tortoise_orm(scene)
            episode_excl.scenes.append(scene_excl)
            if scene_excl.sequence_in_episode in scenes_to_events:
                for i in range(len(scenes_to_events[scene_excl.sequence_in_episode])):
                    scene_event = scenes_to_events[scene_excl.sequence_in_episode][i]
                    scene_event.sequence_in_scene = i+1
                    # TODO I am STUMPED as to why scene_event.id must be set while scene.id and episode.id do not
                    scene_event.id = i+1
                    # print(f'@@@@@@@@ scene_event={scene_event}')
                    # print(f'@@@@@@@@ vars(scene_event)={vars(scene_event)}')                    
                    # scene_event.scene = scene_excl
                    scene_event_excl = await Scene_Event_Pydantic_Excluding.from_tortoise_orm(scene_event)
                    scene_excl.events.append(scene_event_excl)
        return {'show': show_metadata[show_key], 'raw_episode': raw_episode, 'episode': episode_excl}


# @transcript_playground_app.get("/load_transcripts/{show_key}/{transcript_type}")
# async def load_transcripts(show_key: Show, transcript_type: TranscriptType|None = None):
#     json_episodes = import_transcripts_by_type(show_key, transcript_type) 
#     print(f'loaded transcript count={len(json_episodes)}, transcript episodes={json_episodes.keys()}')
#     return {"loaded transcript count": len(json_episodes), "loaded transcript episodes": list(json_episodes.keys())}


@transcript_playground_app.get("/episode/{show_key}/{episode_key}")
async def fetch_episode(show_key: ShowKey, episode_key: str):
    try:
        episode = await dao.fetch_episode(show_key.value, episode_key)
    except Exception as e:
        return {"Error": f"Failure to fetch Episode having show_key={show_key} episode_key={episode_key}: {e}"}
    return await Episode_Pydantic.from_tortoise_orm(episode)





########### BEGIN EXAMPLES #############
# https://medium.com/@talhakhalid101/python-tortoise-orm-integration-with-fastapi-c3751d248ce1

@transcript_playground_app.post("/job/create/", status_code=201)
# async def create_job(name=Form(...), description=Form(...)):
async def create_job(name, description):
    job = await Job.create(name=name, description=description)
    return await job_pydantic.from_tortoise_orm(job)

@transcript_playground_app.get("/job/{job_id}", response_model=job_pydantic, responses={404: {"model": HTTPNotFoundError}})
async def get_job(job_id: int):
    return await job_pydantic_no_ids.from_queryset_single(Job.get(id=job_id))

@transcript_playground_app.get("/jobs/")
async def get_jobs():
    return await job_pydantic.from_queryset(Job.all())

# TODO this doesn't work
@transcript_playground_app.put("/job/{job_id}", response_model=job_pydantic, responses={404: {"model": HTTPNotFoundError}})
async def update_job(job_id: int, job: job_pydantic_no_ids):
    res = Job.filter(id=job_id)
    print(f'fetched job={job}')
    await res.update(**job.dict())
    # await Job.filter(id=job_id).update(**job.dict())
    return await job_pydantic_no_ids.from_queryset_single(Job.get(id=job_id))

@transcript_playground_app.delete("/job/{job_id}", response_model=Status, responses={404: {"model": HTTPNotFoundError}})
async def delete_job(job_id: int):
    deleted_job = await Job.filter(id=job_id).delete()
    if not deleted_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return Status(message=f"Deleted job {job_id}")


########### OLDER EXAMPLES #############

# @transcript_playground_app.get("/item/{item_id}")
# async def read_item(item_id: int):
#     return {"item_id": item_id}


########### END EXAMPLES #############

