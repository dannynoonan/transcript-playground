from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from operator import itemgetter
import requests

import main
from show_metadata import ShowKey


templates = Jinja2Templates(directory="templates")
web_app = APIRouter()
# web_app.mount('/static', StaticFiles(directory='static', html=True), name='static')


@web_app.get("/web/show/{show_key}")
async def home(request: Request, show_key: ShowKey):
	tdata = {}

	tdata['header'] = 'show'
	tdata['show_key'] = show_key.value

	locations_by_scene = await main.agg_scenes_by_location(show_key)
	tdata['locations_by_scene'] = locations_by_scene['scenes_by_location']

	speaker_counts = await main.composite_speaker_aggs(show_key)
	tdata['speaker_counts'] = speaker_counts['speaker_agg_composite']

	keywords = await main.keywords_by_corpus(show_key, exclude_speakers=True)
	tdata['keywords'] = keywords['keywords']

	# season_count = 7
	# for i in range(season_count):
	# 	s = i+1
	# 	season_word_counts = await main.agg_dialog_word_counts(show_key, season=s)
	# 	season_location_counts = await main.agg_scenes_by_location(show_key, season=s)

	# 	season_speaker_episode_counts = await main.agg_episodes_by_speaker(show_key, season=s)
	# 	season_speaker_scene_counts = await main.agg_scenes_by_speaker(show_key, season=s)
	# 	season_speaker_line_counts = await main.agg_scene_events_by_speaker(show_key, season=2)
	# 	season_speaker_word_counts = await main.agg_dialog_word_counts(show_key, season=s)

	# 	season_keywords = await main.keywords_by_corpus(show_key, exclude_speakers=True)

	return templates.TemplateResponse("show.html", {"request": request, 'tdata': tdata})


@web_app.get("/web/episode/{show_key}/{episode_key}", response_class=HTMLResponse)
async def episode_page(request: Request, show_key: ShowKey, episode_key: str):
	tdata = {}

	tdata['header'] = 'episode'
	tdata['show_key'] = show_key.value
	tdata['episode_key'] = episode_key
	
	episode = await main.fetch_episode(show_key, episode_key, data_source='es')
	tdata['episode'] = episode['es_episode']
	
	locations_by_scene = await main.agg_scenes_by_location(show_key, episode_key=episode_key)
	tdata['locations_by_scene'] = locations_by_scene['scenes_by_location']

	episode_word_counts = await main.agg_dialog_word_counts(show_key, episode_key=episode_key)
	tdata['episode_word_counts'] = episode_word_counts['dialog_word_counts']

	speaker_counts = await main.composite_speaker_aggs(show_key, episode_key=episode_key)
	tdata['speaker_counts'] = speaker_counts['speaker_agg_composite']

	# speaker_scene_counts = await main.agg_scenes_by_speaker(show_key, episode_key=episode_key)
	# speaker_line_counts = await main.agg_scene_events_by_speaker(show_key, episode_key=episode_key)
	# speaker_word_counts = await main.agg_dialog_word_counts(show_key, episode_key=episode_key) 
	#
	# speakers = {}
	# for speaker, scene_count in speaker_scene_counts['scenes_by_speaker'].items():
	# 	speakers[speaker] = {} 
	# 	speakers[speaker]['speaker'] = speaker
	# 	speakers[speaker]['scene_count'] = scene_count
	# for speaker, line_count in speaker_line_counts['scene_events_by_speaker'].items():
	# 	if speaker not in speakers:
	# 		speakers[speaker] = {}
	# 		speakers[speaker]['speaker'] = speaker
	# 	speakers[speaker]['line_count'] = line_count
	# for speaker, word_count in speaker_word_counts['dialog_word_counts'].items():
	# 	if speaker not in speakers:
	# 		speakers[speaker] = {}
	# 		speakers[speaker]['speaker'] = speaker
	# 	speakers[speaker]['word_count'] = int(word_count)  # NOTE not sure why casting is needed here
	#
	# speaker_dicts = speakers.values()
	# tdata['speaker_counts'] = sorted(speaker_dicts, key=itemgetter('scene_count'), reverse=True)
	
	keywords = await main.keywords_by_episode(show_key, episode_key, exclude_speakers=True)
	tdata['keywords'] = keywords['keywords']
	
	mlt = await main.more_like_this(show_key, episode_key)
	tdata['similar_episodes'] = mlt['similar_episodes']
	
	return templates.TemplateResponse('episode.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/episode_search/{show_key}", response_class=HTMLResponse)
async def episode_search(request: Request, show_key: ShowKey, season: str = None, qt: str = None):
	tdata = {}

	tdata['header'] = 'episode'
	tdata['show_key'] = show_key.value
	tdata['season'] = season

	if not qt:
		tdata['qt'] = ''

	else:
		tdata['qt'] = qt
		episode_matches = await main.search(show_key, season=season, qt=qt)
		tdata['episode_matches'] = episode_matches['matches']
		tdata['episode_match_count'] = episode_matches['episode_count']

	return templates.TemplateResponse('episodeSearch.html', {'request': request, 'tdata': tdata})
