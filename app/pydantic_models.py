from tortoise import Tortoise
from tortoise.contrib.pydantic import pydantic_model_creator

from app.models import Episode, TranscriptSource, Scene, SceneEvent


Tortoise.init_models(["app.models"], "models")


TranscriptSourcePydantic = pydantic_model_creator(TranscriptSource)
EpisodePydantic = pydantic_model_creator(Episode)
ScenePydantic = pydantic_model_creator(Scene)
SceneEventPydantic = pydantic_model_creator(SceneEvent)

TranscriptSourcePydanticExcluding = pydantic_model_creator(TranscriptSource, exclude=("id", "episode", "loaded_ts"))
EpisodePydanticExcluding = pydantic_model_creator(Episode, exclude=("id", "loaded_ts", "transcript_loaded_ts"))
ScenePydanticExcluding = pydantic_model_creator(Scene, exclude=("id", "episode", "episode_id"))
SceneEventPydanticExcluding = pydantic_model_creator(SceneEvent, exclude=("id", "scene", "scene_id"))
