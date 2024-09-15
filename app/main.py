from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from tortoise.contrib.fastapi import HTTPNotFoundError, register_tortoise
from tortoise import Tortoise

from app.config import settings, DATABASE_URL
from app.dash_app import dapp
from app.dash_app_new import dapp_new
from app.database.connect import connect_to_database
import app.database.dao as dao
from app.es.es_read_router import esr_app
from app.es.es_write_router import esw_app
from app.etl.etl_router import etl_app
import app.pydantic_models as pymod
from app.show_metadata import ShowKey, show_metadata
from app.web.web_router import web_app


app = FastAPI()
app.include_router(web_app)
app.include_router(etl_app)
app.include_router(esw_app)
app.include_router(esr_app)
app.mount('/static', StaticFiles(directory='static', html=True), name='static')
app.mount('/tsp_dash', WSGIMiddleware(dapp.server))
app.mount('/tsp_dash_new', WSGIMiddleware(dapp_new.server))
# templates = Jinja2Templates(directory="templates")


# @app.get("/web2")
# async def home(request: Request):
# 	return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/web2/episode/{show_key}/{episode_key}", response_class=HTMLResponse)
# async def fetch_episode(request: Request, show_key: str, episode_key: str):
#     return templates.TemplateResponse('episode.html', {'request': request, 'show_key': show_key, 'episode_key': episode_key})


register_tortoise(
    app,
    # db_url="sqlite://db.sqlite3",
    db_url=DATABASE_URL,
    modules={"models": ["app.models"]},
    generate_schemas=True,
    add_exception_handlers=True,
)

# I used to think this duplicated the `register_tortoise` functionality and have never understood how/why
# async def init():
#     # Here we create a SQLite DB using file "db.sqlite3"
#     #  also specify the app name of "models"
#     #  which contain models from "app.models"
#     await Tortoise.init(
#         db_url=DATABASE_URL,
#         modules={'models': ['app.models']}
#     )
#     # Generate the schema
#     await Tortoise.generate_schemas()


# TODO pretty sure this can be removed
Tortoise.init_models(["app.models"], "models")


@app.get("/", tags=['Admin'])
def root():
    return {"message": "Welcome to transcript playground"}



###################### METADATA ###########################

@app.get("/show_meta/{show_key}", tags=['Metadata'])
async def fetch_show_meta(show_key: ShowKey):
    show_meta = show_metadata[show_key]
    return {show_key: show_meta}



###################### DB ADMIN ###########################

@app.get("/db_connect", tags=['Admin'])
async def db_connect():
    await connect_to_database()
    return {"DB connection": "Indeed"}


@app.get("/backup_db", tags=['Admin'])
async def backup_db():
    await connect_to_database()
    output, error = await dao.backup_db()
    return {"Output": str(output), "Error": str(error)}



###################### DB READ / ID-BASED LOOKUP ###########################

@app.get("/db_episode/{show_key}/{episode_key}", tags=['DB Reader'])
async def fetch_db_episode(show_key: ShowKey, episode_key: str):
    # fetch episode from db
    episode = None
    try:
        episode = await dao.fetch_episode(show_key.value, episode_key, fetch_related=['scenes', 'events'])
    except Exception as e:
        return {"Error": f"Failure to fetch Episode having show_key={show_key} external_key={episode_key} (have run /load_episode_listing?): {e}"}
    if not episode:
        return {"Error": f"No Episode found having show_key={show_key} external_key={episode_key}. You may need to run /load_episode_listing first."}
    
    episode_pyd = await pymod.EpisodePydantic.from_tortoise_orm(episode)

    # NOTE this generates json versions of pydantic model, not sure where to put this code 
    # episode_json = episode_pyd.model_dump_json()
    # print(f'episode_json={episode_json}')
    # with open(f"episode_{show_key}_{episode_key}.json", "w") as file:
    #     json.dump(episode_json, file, indent=4)

    return {"show_meta": show_metadata[show_key], "episode": episode_pyd}



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

########### END EXAMPLES #############
