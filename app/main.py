from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.staticfiles import StaticFiles
from mangum import Mangum
from tortoise.contrib.fastapi import register_tortoise
from tortoise import Tortoise

from app.config import DATABASE_URL
from app.dash_pages import dash_pages_app
from app.routers.auth_router import auth_app
# from app.routers.db_admin_router import dba_app
# from app.routers.db_read_router import dbr_app
# from app.routers.es_admin_router import esa_app
# from app.routers.es_bulk_write_router import esbw_app
from app.routers.es_read_router import esr_app
from app.routers.es_write_router import esw_app
from app.routers.etl_router import etl_app
from app.routers.web_router import web_app


app = FastAPI()
app.include_router(auth_app)
app.include_router(esr_app)
# app.include_router(dbr_app)
app.include_router(web_app)
app.include_router(esw_app)
# app.include_router(esbw_app)
app.include_router(etl_app)
# app.include_router(dba_app)
# app.include_router(esa_app)
app.mount('/static', StaticFiles(directory='static', html=True), name='static')
app.mount('/dash_pages', WSGIMiddleware(dash_pages_app.server))


# NOTE was I using this during a brief fling with Lambda deployment?
handler = Mangum(app)


register_tortoise(
    app,
    db_url=DATABASE_URL,
    modules={"models": ["app.models"]},
    generate_schemas=True,
    add_exception_handlers=True,
)

# I used to think this duplicated the `register_tortoise` functionality and have never understood how/why
# async def init():
#     await Tortoise.init(
#         db_url=DATABASE_URL,
#         modules={'models': ['app.models']}
#     )
#     # Generate the schema
#     await Tortoise.generate_schemas()

# TODO pretty sure this can be removed
Tortoise.init_models(["app.models"], "models")




###################### METADATA ###########################

# @app.get("/show_meta/{show_key}", tags=['Metadata'])
# def fetch_show_meta(show_key: ShowKey, user: user_dependency):
#     exit_if_unauthorized(user, level='admin')
#     show_meta = show_metadata[show_key]
#     return {show_key: show_meta}



# @app.get("/web2")
# async def home(request: Request):
# 	return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/web2/episode/{show_key}/{episode_key}", response_class=HTMLResponse)
# async def fetch_episode(request: Request, show_key: str, episode_key: str):
#     return templates.TemplateResponse('episode.html', {'request': request, 'show_key': show_key, 'episode_key': episode_key})



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
