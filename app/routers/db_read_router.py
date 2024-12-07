from fastapi import APIRouter

from app.auth import user_dependency, exit_if_unauthorized
from app.database import dao
import app.pydantic_models as pymod
from app.show_metadata import ShowKey


dbr_app = APIRouter(prefix='/dbr', tags=['DB Reader'])


@dbr_app.get("/db_episode/{show_key}/{episode_key}")
async def fetch_db_episode(show_key: ShowKey, episode_key: str, user: user_dependency):
    exit_if_unauthorized(user, level='admin')

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

    return {"episode": episode_pyd}
