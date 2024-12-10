from pydantic import BaseModel
from tortoise import Tortoise
from tortoise.contrib.pydantic import pydantic_model_creator

from app.models import Episode, TranscriptSource, Scene, SceneEvent

# NOTE first time in here for a while, but seems surprising/redundant that this is here
Tortoise.init_models(["app.models"], "models")

# NOTE these X_pydantic objects are hella old, not fully understood, and probably a bit irrelevant
# https://docs.pydantic.dev/latest/api/config/
# https://tortoise.github.io/contrib/pydantic.html

TranscriptSourcePydantic = pydantic_model_creator(TranscriptSource)
EpisodePydantic = pydantic_model_creator(Episode)
ScenePydantic = pydantic_model_creator(Scene)
SceneEventPydantic = pydantic_model_creator(SceneEvent)
# APIUserPydantic = pydantic_model_creator(APIUser)

TranscriptSourcePydanticExcluding = pydantic_model_creator(TranscriptSource, exclude=("id", "episode", "loaded_ts"))
EpisodePydanticExcluding = pydantic_model_creator(Episode, exclude=("id", "loaded_ts", "transcript_loaded_ts"))
ScenePydanticExcluding = pydantic_model_creator(Scene, exclude=("id", "episode", "episode_id"))
SceneEventPydanticExcluding = pydantic_model_creator(SceneEvent, exclude=("id", "scene", "scene_id"))

# EpisodePydanticExcludingMore = pydantic_model_creator(Episode, exclude=("id", "loaded_ts", "transcript_loaded_ts", "scenes", "transcript_sources"))

class CreateUserRequest(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
