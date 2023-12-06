from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from operator import itemgetter
from tortoise.contrib.fastapi import HTTPNotFoundError, register_tortoise
from tortoise.contrib.pydantic import pydantic_model_creator
from tortoise import Tortoise

from app.models import TranscriptSource, Episode, Scene, SceneEvent
from config import settings, DATABASE_URL
import dao
from database.connect import connect_to_database
from es_ingest_transformer import to_es_episode
import es_query_builder as esqb
import es_response_transformer as esrt
from show_metadata import ShowKey, Status, show_metadata
from soup_brewer import get_episode_detail_listing_soup, get_transcript_url_listing_soup, get_transcript_soup
from transcript_extractor import parse_episode_transcript_soup
from transcript_listing_extractor import parse_episode_listing_soup, parse_transcript_url_listing_soup, match_episodes_to_transcript_urls
from web.web_router import web_app


app = FastAPI()
app.include_router(web_app)
app.mount('/static', StaticFiles(directory='static', html=True), name='static')


# templates = Jinja2Templates(directory="templates")

# @app.get("/web2")
# async def home(request: Request):
# 	return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/web2/episode/{show_key}/{episode_key}", response_class=HTMLResponse)
# async def fetch_episode(request: Request, show_key: str, episode_key: str):
#     return templates.TemplateResponse('episode.html', {'request': request, 'show_key': show_key, 'episode_key': episode_key})


# https://fastapi.tiangolo.com/advanced/settings/#__tabbed_2_1
# @lru_cache
# def get_settings():
#     return Settings()


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
    app,
    # db_url="sqlite://db.sqlite3",
    db_url=DATABASE_URL,
    modules={"models": ["app.models"]},
    generate_schemas=True,
    add_exception_handlers=True,
)
##### END THESE DO THE SAME THING ######

Tortoise.init_models(["app.models"], "models")

# https://docs.pydantic.dev/latest/api/config/
# https://tortoise.github.io/contrib/pydantic.html

# JobPydantic = pydantic_model_creator(Job)
# JobPydanticNoIds = pydantic_model_creator(Job, exclude_readonly=True)

TranscriptSourcePydantic = pydantic_model_creator(TranscriptSource)
EpisodePydantic = pydantic_model_creator(Episode)
ScenePydantic = pydantic_model_creator(Scene)
SceneEventPydantic = pydantic_model_creator(SceneEvent)

TranscriptSourcePydanticExcluding = pydantic_model_creator(TranscriptSource, exclude=("id", "episode", "loaded_ts"))
EpisodePydanticExcluding = pydantic_model_creator(Episode, exclude=("id", "loaded_ts", "transcript_loaded_ts"))
ScenePydanticExcluding = pydantic_model_creator(Scene, exclude=("id", "episode", "episode_id"))
SceneEventPydanticExcluding = pydantic_model_creator(SceneEvent, exclude=("id", "scene", "scene_id"))

# print(f'Episode_Pydantic.model_json_schema()={Episode_Pydantic.model_json_schema()}')



# https://fastapi.tiangolo.com/tutorial/query-params-str-validations/

@app.get("/")
def root():
    return {"message": "Welcome to transcript playground"}


@app.get("/db_connect")
async def db_connect():
    await connect_to_database()
    return {"DB connection": "Indeed"}


@app.get("/show_meta/{show_key}")
async def fetch_show_meta(show_key: ShowKey):
    show_meta = show_metadata[show_key]
    return {show_key: show_meta}


@app.get("/load_episode_listing/{show_key}")
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
            episode_excl = await EpisodePydanticExcluding.from_tortoise_orm(episode)
            episodes_excl.append(episode_excl)
        return {'episode_count': len(episodes_excl), 'episodes': episodes_excl}


@app.get("/load_transcript_sources/{show_key}")
async def load_transcript_sources(show_key: ShowKey, write_to_db: bool = False):
    listing_soup = await get_transcript_url_listing_soup(show_key)
    episode_transcripts_by_type = await parse_transcript_url_listing_soup(show_key, listing_soup)
    transcript_sources = await match_episodes_to_transcript_urls(show_key, episode_transcripts_by_type)
    if write_to_db:
        stored_tx_sources = []
        for tx_source in transcript_sources:
            stored_tx_source = await dao.upsert_transcript_source(tx_source)
            stored_tx_sources.append(stored_tx_source)
        return {'transcript_sources_count': len(stored_tx_sources), 'transcript_sources': stored_tx_sources}
    else:
        return {'transcript_sources_count': len(transcript_sources), 'transcript_sources': transcript_sources}
    

@app.get("/load_transcript/{show_key}/{episode_key}")
async def load_transcript(show_key: ShowKey, episode_key: str, write_to_db: bool = False):
    episode = None
    # fetch episode and transcript_source(s), throw errors if not found
    try:
        episode = await dao.fetch_episode(show_key.value, episode_key)
    except Exception as e:
        return {"Error": f"Failure to fetch Episode having show_key={show_key} external_key={episode_key} (have run /load_episode_listing?): {e}"}
    if not episode:
        return {"Error": f"No Episode found having show_key={show_key} external_key={episode_key}. You may need to run /load_episode_listing first."}
    await episode.fetch_related('transcript_sources')
    if not episode.transcript_sources:
        return {"Error": f"No Transcript found for episode having show_key={show_key} external_key={episode_key}. You may need to run /load_transcript_sources first."}
    
    # fetch and transform raw transcript into persistable scene and scene_event data
    # TODO data model permits multiple transcript_sources per episode, for now just choose first one
    transcript_source = episode.transcript_sources[0]
    transcript_soup = await get_transcript_soup(episode.transcript_sources[0])
    scenes, scenes_to_events = await parse_episode_transcript_soup(episode, transcript_source.transcript_type, transcript_soup)
    
    if write_to_db:
        await dao.insert_transcript(episode, scenes=scenes, scenes_to_events=scenes_to_events)
        episode_pyd = await EpisodePydantic.from_tortoise_orm(episode)
        return {'show': show_metadata[show_key], 'episode': episode_pyd}
    else:
        # this response should be identical to the persisted version above
        episode_excl = await EpisodePydanticExcluding.from_tortoise_orm(episode)
        for scene in scenes:
            scene_excl = await ScenePydanticExcluding.from_tortoise_orm(scene)
            episode_excl.scenes.append(scene_excl)
            if scene_excl.sequence_in_episode in scenes_to_events:
                for i in range(len(scenes_to_events[scene_excl.sequence_in_episode])):
                    scene_event = scenes_to_events[scene_excl.sequence_in_episode][i]
                    scene_event.sequence_in_scene = i+1
                    # TODO I am STUMPED as to why scene_event.id must be set while scene.id and episode.id do not
                    scene_event.id = i+1
                    scene_event_excl = await SceneEventPydanticExcluding.from_tortoise_orm(scene_event)
                    scene_excl.events.append(scene_event_excl)
        return {'show': show_metadata[show_key], 'episode': episode_excl}


@app.get("/load_all_transcripts/{show_key}")
async def load_all_transcripts(show_key: ShowKey, overwrite_all: bool = False):
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
            print(f"No Transcript found for episode having show_key={show_key} external_key={episode.external_key}. You may need to run /load_transcript_sources first.")
            no_transcript_episode_keys.append(episode.external_key)
            continue
        # fetch and transform raw transcript into persistable scene and scene_event data
        # TODO data model permits multiple transcript_sources per episode, for now just choose first one
        transcript_source = episode.transcript_sources[0]
        transcript_soup = await get_transcript_soup(episode.transcript_sources[0])
        scenes, scenes_to_events = await parse_episode_transcript_soup(episode, transcript_source.transcript_type, transcript_soup)
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


@app.get("/episode/{show_key}/{episode_key}")
async def fetch_episode(show_key: ShowKey, episode_key: str, data_source: str = None):
    if not data_source:
        data_source = 'db'

    # fetch episode from es
    if data_source == 'es':
        s = await esqb.fetch_episode_by_key(show_key.value, episode_key)
        es_query = s.to_dict()
        match = await esrt.return_episode_by_key(s)
        return {"es_episode": match, 'es_query': es_query}
    
    # fetch episode from db
    episode = None
    try:
        episode = await dao.fetch_episode(show_key.value, episode_key)
    except Exception as e:
        return {"Error": f"Failure to fetch Episode having show_key={show_key} external_key={episode_key} (have run /load_episode_listing?): {e}"}
    if not episode:
        return {"Error": f"No Episode found having show_key={show_key} external_key={episode_key}. You may need to run /load_episode_listing first."}
    
    # fetch nested scene and scene_event data
    await episode.fetch_related('scenes')
    for scene in episode.scenes:
        await scene.fetch_related('events')
    
    episode_pyd = await EpisodePydantic.from_tortoise_orm(episode)
    return {"show_meta": show_metadata[show_key], "episode": episode_pyd}


@app.get("/init_es")
async def init_es():
    await esqb.init_mappings()
    return {"success": "success"}


@app.get("/index_episode/{show_key}/{episode_key}")
async def index_transcript(show_key: ShowKey, episode_key: str):
    # fetch episode, throw errors if not found
    episode = None
    try:
        episode = await dao.fetch_episode(show_key.value, episode_key)
    except Exception as e:
        return {"Error": f"Failure to fetch Episode having show_key={show_key} external_key={episode_key} (have run /load_episode_listing?): {e}"}
    if not episode:
        return {"Error": f"No Episode found having show_key={show_key} external_key={episode_key}. You may need to run /load_episode_listing first."}
    
    # fetch nested scene and scene_event data
    await episode.fetch_related('scenes')
    for scene in episode.scenes:
        await scene.fetch_related('events')
    
    # transform to es-writable object and write to es
    try:
        es_episode = await to_es_episode(episode)
        await esqb.save_es_episode(es_episode)
    except Exception as e:
        return {"Error": f"Failure to transform Episode {show_key}:{episode_key} to es-writable version: {e}"}

    return {"Success": f"Episode {show_key}:{episode_key} written to es index"}


@app.get("/index_all_episodes/{show_key}")
async def index_all_transcripts(show_key: ShowKey, overwrite_all: bool = False):
    episodes = []
    try:
        episodes = await dao.fetch_episodes(show_key.value)
    except Exception as e:
        return {"Error": f"Failure to fetch Episodes having show_key={show_key}: {e}"}
    if not episodes:
        return {"Error": f"No Episodes found having show_key={show_key}. You may need to run /load_episode_listing first."}
    if not overwrite_all:
        return {"No-op": f"/index_transcripts was invoked on {len(episodes)} episodes, but `overwrite_all` flag was not set to True so no action was taken"}
    
    # fetch and insert transcripts for all episodes
    attempts = 0
    successful_episode_keys = []
    failed_episode_keys = []
    for episode in episodes:
        attempts += 1

        # fetch nested scene and scene_event data
        await episode.fetch_related('scenes')
        # if not episode.scenes:
        #     print(f"No Scene data found for episode {show_key}_{episode.external_key}. You may need to run /load_transcript first.")
        #     failed_episode_keys.append(episode.external_key)
        #     continue
        for scene in episode.scenes:
            await scene.fetch_related('events')

        # transform to es-writable object and write to es
        try:
            es_episode = await to_es_episode(episode)
            await esqb.save_es_episode(es_episode)
            successful_episode_keys.append(episode.external_key)
        except Exception as e:
            failed_episode_keys.append(episode.external_key)
            print(f"Failure to transform Episode {show_key}_{episode.external_key} to es-writable version or write it to es: {e}")

    return {
        "index loading attempts": attempts, 
        "successful": len(successful_episode_keys),
        "successful episode keys": successful_episode_keys, 
        "failed": len(failed_episode_keys),
        "failed episode keys": failed_episode_keys, 
    }


@app.get("/search_doc_ids/{show_key}")
async def search_doc_ids(show_key: ShowKey, season: str = None):
    s = await esqb.search_doc_ids(show_key.value, season=season)
    es_query = s.to_dict()
    matches = await esrt.return_doc_ids(s)
    return {"doc_count": len(matches), "doc_ids": matches, "es_query": es_query}


@app.get("/search_episodes_by_title/{show_key}")
async def search_episodes_by_title(show_key: ShowKey, title: str = None):
    s = await esqb.search_episodes_by_title(show_key.value, title)
    es_query = s.to_dict()
    matches = await esrt.return_episodes_by_title(s)
    return {"episode_count": len(matches), "episodes": matches, "es_query": es_query}


@app.get("/search_scenes/{show_key}")
async def search_scenes(show_key: ShowKey, season: str = None, episode_key: str = None, location: str = None, description: str = None):
    if not (location or description):
        error = 'Unable to execute search_scenes without at least one scene property set (location or description)'
        print(error)
        return {"error": error}
    s = await esqb.search_scenes(show_key.value, season=season, episode_key=episode_key, location=location, description=description)
    es_query = s.to_dict()
    matches, scene_count = await esrt.return_scenes(s)
    return {"episode_count": len(matches), "scene_count": scene_count, "matches": matches, "es_query": es_query}


@app.get("/search_scene_events/{show_key}")
async def search_scene_events(show_key: ShowKey, season: str = None, episode_key: str = None, speaker: str = None, dialog: str = None, location: str = None):
    s = await esqb.search_scene_events(show_key.value, season=season, episode_key=episode_key, speaker=speaker, dialog=dialog)
    es_query = s.to_dict()
    matches, scene_count, scene_event_count = await esrt.return_scene_events(s, location=location)
    return {"episode_count": len(matches), "scene_count": scene_count, "scene_event_count": scene_event_count, "matches": matches, "es_query": es_query}


@app.get("/search_scene_events_multi_speaker/{show_key}")
async def search_scene_events_multi_speaker(show_key: ShowKey, season: str = None, episode_key: str = None, speakers: str = None, location: str = None):
    s = await esqb.search_scene_events_multi_speaker(show_key.value, speakers, season=season, episode_key=episode_key)
    es_query = s.to_dict()
    matches, scene_count, scene_event_count = await esrt.return_scene_events_multi_speaker(s, speakers, location=location)
    return {"episode_count": len(matches), "scene_count": scene_count, "scene_event_count": scene_event_count, "matches": matches, "es_query": es_query}


@app.get("/search/{show_key}")
async def search(show_key: ShowKey, season: str = None, episode_key: str = None, qt: str = None):
    s = await esqb.search_episodes(show_key.value, season=season, episode_key=episode_key, qt=qt)
    es_query = s.to_dict()
    matches, scene_count, scene_event_count = await esrt.return_episodes(s)
    return {"episode_count": len(matches), "scene_count": scene_count, "scene_event_count": scene_event_count, "matches": matches, "es_query": es_query}


@app.get("/agg_episodes/{show_key}")
async def agg_episodes(show_key: ShowKey, season: str = None, location: str = None):
    s = await esqb.agg_episodes(show_key.value, season=season, location=location)
    es_query = s.to_dict()
    episode_count = await esrt.return_episode_count(s)
    return {"episode_count": episode_count, "es_query": es_query}


@app.get("/agg_episodes_by_speaker/{show_key}")
async def agg_episodes_by_speaker(show_key: ShowKey, season: str = None, location: str = None, other_speaker: str = None):
    s = await esqb.agg_episodes_by_speaker(show_key.value, season=season, location=location, other_speaker=other_speaker)
    es_query = s.to_dict()
    # separate call to get episode_count without double-counting per speaker
    episode_count = await agg_episodes(show_key, season=season, location=location)
    matches = await esrt.return_episodes_by_speaker(s, episode_count['episode_count'], location=location, other_speaker=other_speaker)
    return {"speaker_count": len(matches), "episodes_by_speaker": matches, "es_query": es_query}


@app.get("/agg_scenes_by_location/{show_key}")
async def agg_scenes_by_location(show_key: ShowKey, season: str = None, episode_key: str = None, speaker: str = None):
    s = await esqb.agg_scenes_by_location(show_key.value, season=season, episode_key=episode_key, speaker=speaker)
    es_query = s.to_dict()
    matches = await esrt.return_scenes_by_location(s, speaker=speaker)
    return {"location_count": len(matches), "scenes_by_location": matches, "es_query": es_query}


@app.get("/agg_scenes_by_speaker/{show_key}")
async def agg_scenes_by_speaker(show_key: ShowKey, season: str = None, episode_key: str = None, location: str = None, other_speaker: str = None):
    s = await esqb.agg_scenes_by_speaker(show_key.value, season=season, episode_key=episode_key, location=location, other_speaker=other_speaker)
    es_query = s.to_dict()
    # separate call to get scene_count without double-counting per speaker
    scene_count = await agg_scenes(show_key, season=season, episode_key=episode_key, location=location)
    matches = await esrt.return_scenes_by_speaker(s, scene_count['scene_count'], location=location, other_speaker=other_speaker)
    return {"speaker_count": len(matches), "scenes_by_speaker": matches, "es_query": es_query}


@app.get("/agg_scenes/{show_key}")
async def agg_scenes(show_key: ShowKey, season: str = None, episode_key: str = None, location: str = None):
    s = await esqb.agg_scenes(show_key.value, season=season, episode_key=episode_key, location=location)
    es_query = s.to_dict()
    scene_count = await esrt.return_scene_count(s)
    return {"scene_count": scene_count, "es_query": es_query}


@app.get("/agg_scene_events_by_speaker/{show_key}")
async def agg_scene_events_by_speaker(show_key: ShowKey, season: str = None, episode_key: str = None, dialog: str = None):
    s = await esqb.agg_scene_events_by_speaker(show_key.value, season=season, episode_key=episode_key, dialog=dialog)
    es_query = s.to_dict()
    matches = await esrt.return_scene_events_by_speaker(s, dialog=dialog)
    return {"speaker_count": len(matches), "scene_events_by_speaker": matches, "es_query": es_query}


@app.get("/agg_dialog_word_counts/{show_key}")
async def agg_dialog_word_counts(show_key: ShowKey, season: str = None, episode_key: str = None, speaker: str = None):
    s = await esqb.agg_dialog_word_counts(show_key.value, season=season, episode_key=episode_key, speaker=speaker)
    es_query = s.to_dict()
    matches = await esrt.return_dialog_word_counts(s, speaker=speaker)
    return {"dialog_word_counts": matches, "es_query": es_query}


@app.get("/composite_speaker_aggs/{show_key}")
async def composite_speaker_aggs(show_key: ShowKey, season: str = None, episode_key: str = None):
    if not episode_key:
        speaker_episode_counts = await agg_episodes_by_speaker(show_key, season=season)
    speaker_scene_counts = await agg_scenes_by_speaker(show_key, season=season, episode_key=episode_key)
    speaker_line_counts = await agg_scene_events_by_speaker(show_key, season=season, episode_key=episode_key)
    speaker_word_counts = await agg_dialog_word_counts(show_key, season=season, episode_key=episode_key)

    # TODO refactor this to generically handle dicts threading together
    speakers = {}
    if not episode_key:
        for speaker, episode_count in speaker_episode_counts['episodes_by_speaker'].items():
            if speaker not in speakers:
                speakers[speaker] = {}
                speakers[speaker]['speaker'] = speaker
            speakers[speaker]['episode_count'] = episode_count
    for speaker, scene_count in speaker_scene_counts['scenes_by_speaker'].items():
        if speaker not in speakers:
            speakers[speaker] = {}
            speakers[speaker]['speaker'] = speaker
        speakers[speaker]['scene_count'] = scene_count
    for speaker, line_count in speaker_line_counts['scene_events_by_speaker'].items():
        if speaker not in speakers:
            speakers[speaker] = {}
            speakers[speaker]['speaker'] = speaker
        speakers[speaker]['line_count'] = line_count
    for speaker, word_count in speaker_word_counts['dialog_word_counts'].items():
        if speaker not in speakers:
            speakers[speaker] = {}
            speakers[speaker]['speaker'] = speaker
        speakers[speaker]['word_count'] = int(word_count)  # NOTE not sure why casting is needed here

    # TODO shouldn't I be able to sort on a key for a dict within a dict
    speaker_dicts = speakers.values()
    sort_field = 'scene_count'
    if not episode_key:
        sort_field = 'episode_count'
    speaker_agg_composite = sorted(speaker_dicts, key=itemgetter(sort_field), reverse=True)

    return {"speaker_count": len(speaker_agg_composite), "speaker_agg_composite": speaker_agg_composite} 


@app.get("/keywords_by_episode/{show_key}/{episode_key}")
async def keywords_by_episode(show_key: ShowKey, episode_key: str, exclude_speakers: bool = False):
    response = await esqb.keywords_by_episode(show_key.value, episode_key)
    all_speakers = []
    if exclude_speakers:
        res = await agg_scenes_by_speaker(show_key, episode_key=episode_key) # TODO should this use agg_episodes_by_speaker now?
        all_speakers = res['scenes_by_speaker'].keys()
    matches = await esrt.return_keywords_by_episode(response, exclude_terms=all_speakers)
    return {"keyword_count": len(matches), "keywords": matches}


@app.get("/keywords_by_corpus/{show_key}")
async def keywords_by_corpus(show_key: ShowKey, season: str = None, exclude_speakers: bool = False):
    response = await esqb.keywords_by_corpus(show_key.value, season=season)
    all_speakers = []
    if exclude_speakers:
        res = await agg_episodes_by_speaker(show_key, season=season)
        all_speakers = res['episodes_by_speaker'].keys()
    matches = await esrt.return_keywords_by_corpus(response, exclude_terms=all_speakers)
    return {"keyword_count": len(matches), "keywords": matches}


@app.get("/more_like_this/{show_key}/{episode_key}")
async def more_like_this(show_key: ShowKey, episode_key: str):
    s = await esqb.more_like_this(show_key.value, episode_key)
    es_query = s.to_dict()
    matches = await esrt.return_more_like_this(s)
    return {"similar_episode_count": len(matches), "similar_episodes": matches, "es_query": es_query}


########### BEGIN EXAMPLES #############
# https://medium.com/@talhakhalid101/python-tortoise-orm-integration-with-fastapi-c3751d248ce1

# @transcript_playground_app.post("/job/create/", status_code=201)
# # async def create_job(name=Form(...), description=Form(...)):
# async def create_job(name, description):
#     job = await Job.create(name=name, description=description)
#     return await JobPydantic.from_tortoise_orm(job)

# @transcript_playground_app.get("/job/{job_id}", response_model=JobPydantic, responses={404: {"model": HTTPNotFoundError}})
# async def get_job(job_id: int):
#     return await JobPydanticNoIds.from_queryset_single(Job.get(id=job_id))

# @transcript_playground_app.get("/jobs/")
# async def get_jobs():
#     return await JobPydantic.from_queryset(Job.all())

# # TODO this doesn't work
# @transcript_playground_app.put("/job/{job_id}", response_model=JobPydantic, responses={404: {"model": HTTPNotFoundError}})
# async def update_job(job_id: int, job: JobPydanticNoIds):
#     res = Job.filter(id=job_id)
#     print(f'fetched job={job}')
#     await res.update(**job.dict())
#     # await Job.filter(id=job_id).update(**job.dict())
#     return await JobPydanticNoIds.from_queryset_single(Job.get(id=job_id))

# @transcript_playground_app.delete("/job/{job_id}", response_model=Status, responses={404: {"model": HTTPNotFoundError}})
# async def delete_job(job_id: int):
#     deleted_job = await Job.filter(id=job_id).delete()
#     if not deleted_job:
#         raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
#     return Status(message=f"Deleted job {job_id}")


########### OLDER EXAMPLES #############

# @transcript_playground_app.get("/item/{item_id}")
# async def read_item(item_id: int):
#     return {"item_id": item_id}


########### END EXAMPLES #############

