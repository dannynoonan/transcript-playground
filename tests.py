import asyncio
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.testclient import TestClient
import json
# import pydantic
# from pydantic import BaseModel, Json, ValidationError
import pytest
# from starlette.testclient import TestClient
from tortoise import Tortoise

import app.database.dao as dao
import app.es.es_ingest_transformer as esit
import app.etl.transcript_listing_extractor as tle
from app.main import app
# from app.models import Episode
import app.pydantic_models as pymod
from app.show_metadata import ShowKey


pytest_plugins = ('pytest_asyncio',)

client = TestClient(app)

Tortoise.init_models(["app.models"], "models")


# async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
#     return {"q": q, "skip": skip, "limit": limit}

# async def override_dependency(q: str | None = None):
#     return {"q": q, "skip": 5, "limit": 10}

# class ConstrainedJsonModel(BaseModel):
#     json_obj: Json

# EpisodePydantic = pydantic_model_creator(Episode)


def mock_dao_fetch_episode():
    show_key = 'TNG'
    episode_key = '247'
    episode_file = f'test_data/episode_{show_key}_{episode_key}.json'
    freeze_dried_episode = None
    with open(episode_file) as f:
        freeze_dried_episode = json.load(f)
    # episode = pydantic.parse_file_as(path=episode_file, type_=Episode)
    # episode = EpisodePydantic.parse_file(episode_file)
    # print(f'type(freeze_dried_episode)={type(freeze_dried_episode)} freeze_dried_episode={freeze_dried_episode}')
    # episode_json = ConstrainedJsonModel(json_obj=freeze_dried_episode)
    # print(f'type(freeze_dried_episode)={type(freeze_dried_episode)}')
    # episode = EpisodePydantic.model_validate(episode_json)
    episode = pymod.EpisodePydantic.parse_obj(freeze_dried_episode) 
    # print('-------------------------------------------------')
    # print(f'episode={episode}')
    return episode
    

def mock_save_es_episode():
    pass


app.dependency_overrides[dao.fetch_episode] = mock_dao_fetch_episode
# app.dependency_overrides[episode_listing_path] = f'test_data/source/episode_listings/TNG.html'
# app.dependency_overrides[esqb.save_es_episode] = mock_save_es_episode


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to transcript playground"}


def test_parse_episode_listing_soup():
    # load freeze-dried data
    show_key = 'TNG'
    freeze_dried_response = None
    with open(f'test_data/responses/parse_episode_listing_soup_{show_key}.json') as f:
        freeze_dried_response = json.load(f)
    json_episodes = []
    for freeze_dried_episode in freeze_dried_response:
        # episode_json = json.loads(freeze_dried_episode)
        json_episodes.append(freeze_dried_episode)

    # execute function
    file_path = f'test_data/source/episode_listings/{show_key}.html'
    episode_listing_soup = BeautifulSoup(open(file_path).read(), 'html5lib')
    episodes = tle.parse_episode_listing_soup(ShowKey(show_key), episode_listing_soup)
    assert len(episodes) == 176
    for i in range(len(episodes)):
        assert episodes[i].external_key == json_episodes[i]['external_key']
        assert episodes[i].title == json_episodes[i]['title']
        assert episodes[i].sequence_in_season == json_episodes[i]['sequence_in_season']
        assert episodes[i].season == json_episodes[i]['season']


def test_load_episode_listing():
    # load freeze-dried data
    show_key = 'TNG'
    freeze_dried_response = None
    with open(f'test_data/responses/load_episode_listing_{show_key}.json') as f:
        freeze_dried_response = json.load(f)

    # call endpoint
    with TestClient(app) as client:
        response = client.get(f"/etl/load_episode_listing/{show_key}")
        assert response.status_code == 200
        assert response.json() == freeze_dried_response


def test_load_transcript():
    # load freeze-dried data
    show_key = 'TNG'
    episode_key = '247'
    freeze_dried_response = None
    with open(f'test_data/responses/load_transcript_{show_key}_{episode_key}.json') as f:
        freeze_dried_response = json.load(f)

    # call endpoint
    with TestClient(app) as client:
        response = client.get(f"/etl/load_transcript/{show_key}/{episode_key}")
        assert response.status_code == 200
        assert response.json() == freeze_dried_response


@pytest.mark.asyncio
async def test_to_es_episode():
    # load freeze-dried data
    show_key = 'TNG'
    episode_key = '246'
    freeze_dried_episode = None
    with open(f'test_data/pydantic/episode_{show_key}_{episode_key}.json') as f:
        freeze_dried_episode = json.load(f)

    freeze_dried_es_episode = None
    with open(f'test_data/es/es_episode_{show_key}_{episode_key}.json') as f:
        freeze_dried_es_episode = json.load(f)

    episode_pyd = pymod.EpisodePydantic.model_validate(freeze_dried_episode) 
    es_episode = esit.to_es_episode(episode_pyd)
    es_episode_dict = es_episode.to_dict()

    # swear I solved something like this yesterday but that was a gazillion years ago
    del freeze_dried_es_episode['air_date']
    del es_episode_dict['air_date']

    assert es_episode_dict == freeze_dried_es_episode





# @pytest.mark.asyncio
# async def test_parse_episode_listing_soup_messy():
#     # load freeze-dried data
#     show_key = 'TNG'
#     freeze_dried_response = None
#     with open(f'test_data/responses/parse_episode_listing_soup_{show_key}.json') as f:
#         freeze_dried_response = json.load(f)
#     print(f'type(freeze_dried_response)={type(freeze_dried_response)}')

#     episode_pyd_list = []
#     json_episodes = []
#     with TestClient(app) as client:
#         for freeze_dried_episode in freeze_dried_response:
#             # print(f'type(freeze_dried_episode)={type(freeze_dried_episode)} freeze_dried_episode={freeze_dried_episode}')
#             episode_json = json.loads(freeze_dried_episode)
#             # print(f'type(episode_json)={type(episode_json)} episode_json={episode_json}')
#             json_episodes.append(episode_json)
#             # TODO results in AttributeError: 'str' object has no attribute 'fetch_related'
#             # episode_pyd = await pymod.EpisodePydanticExcluding.from_tortoise_orm(element)
#             episode_pyd = pymod.EpisodePydanticExcluding.model_validate(episode_json) 
#             episode_pyd_list.append(episode_pyd)

#     # execute function
#     file_path = f'test_data/source/episode_listing/{show_key}.html'
#     episode_listing_soup = BeautifulSoup(open(file_path).read(), 'html5lib')
#     episodes = tle.parse_episode_listing_soup(ShowKey(show_key), episode_listing_soup)
#     assert len(episodes) == 176
#     # other_json_episodes = []
#     i = 0
#     for i in range(len(episodes)):
#         assert episodes[i].external_key == json_episodes[i]['external_key']
#         assert episodes[i].title == json_episodes[i]['title']
#         assert episodes[i].sequence_in_season == json_episodes[i]['sequence_in_season']
#         assert episodes[i].season == json_episodes[i]['season']
#     # for ep in episodes:
#         # print(f'type(ep)={type(ep)} ep={ep}')
#         # ep_pyd = pymod.EpisodePydanticExcludingMore.model_validate(ep) 
#         # print(f'type(ep_pyd)={type(ep_pyd)} ep_pyd={ep_pyd}')
#         # json_ep = ep_pyd.model_dump_json()
#         # other_json_episodes.append(json_ep)

#     # print(f'type(episodes)={type(episodes)} episodes={episodes}')
#     # assert other_json_episodes == json_episodes
#     # assert episodes == episode_pyd_list
    