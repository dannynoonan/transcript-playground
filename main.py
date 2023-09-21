from enum import Enum
from fastapi import FastAPI, Response
import json

from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from transcript_importer import import_transcript_by_episode_key, import_transcripts_by_type


transcript_playground_app = FastAPI()


@transcript_playground_app.get("/")
def root():
    return {"message": "Hi there World"}

@transcript_playground_app.get("/foo")
async def root():
    return {"message": "Hola World"}

@transcript_playground_app.get("/item/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


class Show(str, Enum):
    TNG = "TNG"
    GoT = "GoT"
    Succession = "Succession"

class TranscriptType(str, Enum):
    Fanon = "Fanon"
    TOC = "TOC"
    Default = "Default"
    ALL = "ALL"


@transcript_playground_app.get("/show_meta/{show_key}")
async def fetch_show_meta(show_key: Show):
    if show_key is Show.TNG:
        return {"show_key": show_key, "message": "Trekkies!"}

    if show_key.value == "GoT":
        return {"show_key": show_key, "message": "Westeros is calling!"}

    return {"show_key": show_key, "message": "Wambsgans for the triple-play!"}

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@transcript_playground_app.get("/episode/{show_key}/{episode}")
async def fetch_episode(show_key: Show, episode: int):
    return {"show_key": show_key, "episode": episode}


@transcript_playground_app.get("/load_transcripts/{show_key}/{transcript_type}")
async def load_transcripts(show_key: Show, transcript_type: TranscriptType|None = None):
    json_episodes = import_transcripts_by_type(show_key, transcript_type) 
    print(f'loaded transcript count={len(json_episodes)}, transcript episodes={json_episodes.keys()}')
    return {"loaded transcript count": len(json_episodes), "loaded transcript episodes": list(json_episodes.keys())}

@transcript_playground_app.get("/load_transcript/{show_key}/{episode_key}")
async def load_transcript(show_key: Show, episode_key: str):
    json_episodes = import_transcript_by_episode_key(show_key, episode_key) 
    # json_episodes = json.dumps(json_episodes, indent=4, default=str)
    # return JSONResponse(content=jsonable_encoder(json_episodes))
    # return {"loaded transcripts": JSONResponse(content=jsonable_encoder(json_episodes))}

    # return {"loaded transcripts": JSONResponse(content=json_episodes)}
    return {"loaded transcript count": len(json_episodes), "loaded transcript episodes": episode_key}

    # json_str = json.dumps(json_episodes, indent=4, default=str)
    # return Response(content=json_str, media_type='application/json')

