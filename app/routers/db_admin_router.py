from fastapi import APIRouter

from app.auth import user_dependency, exit_if_unauthorized
from app.database import dao
from app.database.connect import connect_to_database


dba_app = APIRouter(prefix='/dba', tags=['Admin'])


# @dba_app.get("/db_connect")
@dba_app.post("/db_connect")
async def db_connect(user: user_dependency):
    exit_if_unauthorized(user, level='admin')

    await connect_to_database()
    return {"DB connection": "Indeed"}


# @dba_app.get("/backup_db")
@dba_app.post("/backup_db")
async def backup_db(user: user_dependency):
    exit_if_unauthorized(user, level='admin')

    await connect_to_database()
    output, error = await dao.backup_db()
    return {"Output": str(output), "Error": str(error)}
