from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests

import main
from show_metadata import ShowKey


templates = Jinja2Templates(directory="templates")
web_app = APIRouter()
# web_app.mount('/static', StaticFiles(directory='static', html=True), name='static')


@web_app.get("/web")
async def home(request: Request):
	return templates.TemplateResponse("index.html", {"request": request})


@web_app.get("/web/episode/{show_key}/{episode_key}", response_class=HTMLResponse)
async def episode_page(request: Request, show_key: ShowKey, episode_key: str):
    tdata = {}
    
    tdata['show_key'] = show_key.value
    tdata['episode_key'] = episode_key
    
    episode = await main.fetch_episode(show_key, episode_key, data_source='es')
    tdata['episode'] = episode['es_episode']
    
    locations_by_scene = await main.agg_scenes_by_location(show_key, episode_key=episode_key)    
    tdata['locations_by_scene'] = locations_by_scene['scenes_by_location']
    
    speakers_by_scene = await main.agg_scenes_by_speaker(show_key, episode_key=episode_key)
    tdata['speakers_by_scene'] = speakers_by_scene['scenes_by_speaker']
    
    speaker_line_counts = await main.agg_scene_events_by_speaker(show_key, episode_key=episode_key)
    tdata['speaker_line_counts'] = speaker_line_counts['scene_events_by_speaker']
    
    keywords = await main.keywords_by_episode(show_key, episode_key, exclude_speakers=True)
    tdata['keywords'] = keywords['keywords']
    
    mlt = await main.search_more_like_this(show_key, episode_key)
    tdata['similar_episodes'] = mlt['similar_episodes']
    
    return templates.TemplateResponse('episode.html', {'request': request, 'tdata': tdata})
