from fastapi import APIRouter, Request, Response, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from operator import itemgetter
import requests

import web.data_viz as dz
import es.es_query_builder as esqb
import es.es_response_transformer as esrt
# from es_metadata import MODEL_TYPES
import main
import nlp.embeddings_factory as ef
from show_metadata import ShowKey
from utils import truncate_dict


templates = Jinja2Templates(directory="templates")
web_app = APIRouter()
# web_app.mount('/static', StaticFiles(directory='static', html=True), name='static')


@web_app.get("/web/show/{show_key}")
async def show_page(request: Request, show_key: ShowKey):
	tdata = {}

	tdata['header'] = 'show'
	tdata['show_key'] = show_key.value

	locations_by_scene = await main.agg_scenes_by_location(show_key)
	tdata['locations_by_scene'] = locations_by_scene['scenes_by_location']

	speaker_counts = await main.composite_speaker_aggs(show_key)
	tdata['speaker_counts'] = speaker_counts['speaker_agg_composite']

	keywords = await main.keywords_by_corpus(show_key, exclude_speakers=True)
	tdata['keywords'] = keywords['keywords']

	episodes_by_season = main.list_episodes_by_season(show_key)
	tdata['episodes_by_season'] = episodes_by_season['episodes_by_season']

	stats_by_season = {}
	for season in tdata['episodes_by_season'].keys():
		season_episode_count = len(tdata['episodes_by_season'][season])
		stats = {}
		season_locations = await main.agg_scenes_by_location(show_key, season=season)
		stats['location_count'] = season_locations['location_count']
		stats['location_counts'] = truncate_dict(season_locations['scenes_by_location'], season_episode_count, 1)
		season_speakers = await main.agg_scene_events_by_speaker(show_key, season=season)
		stats['line_count'] = season_speakers['scene_events_by_speaker']['_ALL_']
		stats['speaker_line_counts'] = truncate_dict(season_speakers['scene_events_by_speaker'], season_episode_count, 1)
		season_speaker_scene_counts = await main.agg_scenes_by_speaker(show_key, season=season)
		stats['scene_count'] = season_speaker_scene_counts['scenes_by_speaker']['_ALL_']
		season_speaker_episode_counts = await main.agg_episodes_by_speaker(show_key, season=season)
		stats['speaker_count'] = season_speaker_episode_counts['speaker_count']	
		season_speaker_word_counts = await main.agg_dialog_word_counts(show_key, season=season)
		stats['word_count'] = int(season_speaker_word_counts['dialog_word_counts']['_ALL_'])
		# generate air_date_range
		first_episode_in_season = tdata['episodes_by_season'][season][0]
		last_episode_in_season = tdata['episodes_by_season'][season][-1]
		stats['air_date_range'] = f"{first_episode_in_season['air_date'][:10]} - {last_episode_in_season['air_date'][:10]}"
		stats_by_season[season] = stats
	tdata['stats_by_season'] = stats_by_season

	return templates.TemplateResponse("show.html", {"request": request, 'tdata': tdata})


@web_app.get("/web/episode/{show_key}/{episode_key}", response_class=HTMLResponse)
async def episode_page(request: Request, show_key: ShowKey, episode_key: str, search_type: str = None, qt: str = None,
					   dialog: str = None, speaker: str = None, location: str = None, speakers: str = None, locationAMS: str = None):
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
	
	keywords = await main.keywords_by_episode(show_key, episode_key, exclude_speakers=True)
	tdata['keywords'] = keywords['keywords']
	
	mlt = await main.more_like_this(show_key, episode_key)
	tdata['similar_episodes'] = mlt['similar_episodes']

	###### IN-EPISODE SEARCH ######

	if not search_type:
		tdata['search_type'] = ''
	else:
		tdata['search_type'] = search_type

	tdata['qt'] = ''

	tdata['dialog'] = ''
	tdata['speaker'] = ''
	tdata['location'] = ''

	tdata['speakers'] = ''
	tdata['locationAMS'] = ''

	if search_type == 'general':
		tdata['qt'] = qt
		matches = await main.search(show_key, episode_key=episode_key, qt=qt)
		tdata['episode_match'] = matches['matches'][0]
		tdata['scene_match_count'] = matches['scene_count']
		tdata['scene_event_match_count'] = matches['scene_event_count']

	elif search_type == 'advanced':
		if dialog:
			tdata['dialog'] = dialog
		if speaker:
			tdata['speaker'] = speaker
		if location:
			tdata['location'] = location
		# location on its own won't fetch scene_events, if only location is set then invoke /search_scenes
		if location and not (dialog or speaker):
			matches = await main.search_scenes(show_key, episode_key=episode_key, location=location)
		else:
			matches = await main.search_scene_events(show_key, episode_key=episode_key, speaker=speaker, dialog=dialog, location=location)
			tdata['scene_event_match_count'] = matches['scene_event_count']
		tdata['episode_match'] = matches['matches'][0]
		tdata['scene_match_count'] = matches['scene_count']
		
	elif search_type == 'advanced_multi_speaker':
		if speakers:
			tdata['speakers'] = speakers
		if locationAMS:
			tdata['locationAMS'] = locationAMS
		matches = await main.search_scene_events_multi_speaker(show_key, episode_key=episode_key, speakers=speakers, location=locationAMS)
		if matches and matches['matches']:
			tdata['episode_match'] = matches['matches'][0]
			tdata['scene_match_count'] = matches['scene_count']
			tdata['scene_event_match_count'] = matches['scene_event_count']

	return templates.TemplateResponse('episode.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/episode_search/{show_key}", response_class=HTMLResponse)
async def episode_search_page(request: Request, show_key: ShowKey, search_type: str = None, season: str = None, qt: str = None, 
							  dialog: str = None, speaker: str = None, location: str = None, qtSemantic: str = None, model_vendor: str = None, 
							  model_version: str = None, speakers: str = None, locationAMS: str = None):
	tdata = {}

	tdata['header'] = 'episode'
	tdata['show_key'] = show_key.value
	if not search_type:
		tdata['search_type'] = ''
	else:
		tdata['search_type'] = search_type
	tdata['season'] = season

	tdata['qt'] = ''

	tdata['dialog'] = ''
	tdata['speaker'] = ''
	tdata['location'] = ''

	tdata['qtSemantic'] = ''
	tdata['model_vendor'] = ''
	tdata['model_version'] = ''

	tdata['speakers'] = ''
	tdata['locationAMS'] = ''

	if not search_type:
		return templates.TemplateResponse('episodeSearch.html', {'request': request, 'tdata': tdata})

	if search_type == 'general':
		tdata['qt'] = qt
		matches = await main.search(show_key, season=season, qt=qt)
		tdata['episode_matches'] = matches['matches']
		tdata['episode_match_count'] = matches['episode_count']
		tdata['scene_match_count'] = matches['scene_count']
		tdata['scene_event_match_count'] = matches['scene_event_count']

	elif search_type == 'advanced':
		if dialog:
			tdata['dialog'] = dialog
		if speaker:
			tdata['speaker'] = speaker
		if location:
			tdata['location'] = location
		# location on its own won't fetch scene_events, if only location is set then invoke /search_scenes
		if location and not (dialog or speaker):
			matches = await main.search_scenes(show_key, season=season, location=location)
		else:
			matches = await main.search_scene_events(show_key, season=season, speaker=speaker, dialog=dialog, location=location)
			tdata['scene_event_match_count'] = matches['scene_event_count']
		tdata['episode_matches'] = matches['matches']
		tdata['episode_match_count'] = matches['episode_count']
		tdata['scene_match_count'] = matches['scene_count']

	elif search_type == 'semantic':
		tdata['qtSemantic'] = qtSemantic
		if not model_vendor:
			model_vendor = 'webvectors'
		if not model_version:
			model_version = '29'
		tdata['model_vendor'] = model_vendor
		tdata['model_version'] = model_version
		matches = main.vector_search(show_key, qt=qtSemantic, model_vendor=model_vendor, model_version=model_version)
		tdata['episode_matches'] = matches['matches']
		tdata['episode_match_count'] = len(matches['matches'])
		tdata['tokens_processed_count'] = matches['tokens_processed_count']
		tdata['tokens_failed_count'] = matches['tokens_failed_count']
		if 'tokens_processed' in matches:
			tdata['tokens_processed'] = matches['tokens_processed']
		if 'tokens_failed' in matches:
			tdata['tokens_failed'] = matches['tokens_failed']
		
	elif search_type == 'advanced_multi_speaker':
		if speakers:
			tdata['speakers'] = speakers
		if locationAMS:
			tdata['locationAMS'] = locationAMS
		matches = await main.search_scene_events_multi_speaker(show_key, season=season, speakers=speakers, location=locationAMS)
		tdata['episode_matches'] = matches['matches']
		tdata['episode_match_count'] = matches['episode_count']
		tdata['scene_match_count'] = matches['scene_count']
		tdata['scene_event_match_count'] = matches['scene_event_count']

	else:
		print(f'unsupported search_type={search_type}')

	return templates.TemplateResponse('episodeSearch.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/character/{show_key}/{speaker}", response_class=HTMLResponse)
async def character_page(request: Request, show_key: ShowKey, speaker: str, search_type: str = None, season: str = None, 
							  dialog: str = None, location: str = None, speakers: str = None, locationAMS: str = None):
	tdata = {}

	tdata['header'] = 'character'
	tdata['show_key'] = show_key.value
	tdata['speaker'] = speaker

	episode_matches = await main.search_scene_events(show_key, speaker=speaker)
	tdata['episodes'] = episode_matches['matches']
	tdata['episode_count'] = episode_matches['episode_count']
	tdata['scene_count'] = episode_matches['scene_count']
	tdata['scene_event_count'] = episode_matches['scene_event_count']

	word_count = await main.agg_dialog_word_counts(show_key, speaker=speaker)
	tdata['word_count'] = int(word_count['dialog_word_counts'][speaker])
	
	locations_counts = await main.agg_scenes_by_location(show_key, speaker=speaker)
	tdata['location_counts'] = locations_counts['scenes_by_location']

	co_occ_speakers_by_episode = await main.agg_episodes_by_speaker(show_key, other_speaker=speaker)
	co_occ_speakers_by_scene = await main.agg_scenes_by_speaker(show_key, other_speaker=speaker)
	# TODO refactor this to generically handle dicts threading together
	other_speakers = {}
	for other_speaker, episode_count in co_occ_speakers_by_episode['episodes_by_speaker'].items():
		if other_speaker not in other_speakers:
			other_speakers[other_speaker] = {}
			other_speakers[other_speaker]['other_speaker'] = other_speaker
		other_speakers[other_speaker]['episode_count'] = episode_count
	for other_speaker, scene_count in co_occ_speakers_by_scene['scenes_by_speaker'].items():
		if other_speaker not in other_speakers:
			other_speakers[other_speaker] = {}
			other_speakers[other_speaker]['other_speaker'] = other_speaker
		other_speakers[other_speaker]['scene_count'] = scene_count
	del(other_speakers[speaker])
	del(other_speakers['_ALL_'])

	# TODO shouldn't I be able to sort on a key for a dict within a dict
	speaker_dicts = other_speakers.values()
	tdata['other_speaker_agg_composite'] = sorted(speaker_dicts, key=itemgetter('episode_count'), reverse=True)

	###### CHARACTER-CENTRIC SEARCH ######

	if not search_type:
		tdata['search_type'] = ''
	else:
		tdata['search_type'] = search_type

	tdata['dialog'] = ''
	tdata['location'] = ''

	tdata['other_speakers'] = ''
	tdata['locationAMS'] = ''

	if search_type == 'advanced':
		if dialog:
			tdata['dialog'] = dialog
		if location:
			tdata['location'] = location
		matches = await main.search_scene_events(show_key, speaker=speaker, dialog=dialog, location=location)
		tdata['episode_matches'] = matches['matches']
		tdata['episode_match_count'] = matches['episode_count']
		tdata['scene_match_count'] = matches['scene_count']
		tdata['scene_event_match_count'] = matches['scene_event_count']
		
	elif search_type == 'advanced_multi_speaker':
		all_speakers = speaker
		if speakers:
			tdata['speakers'] = speakers
			all_speakers = f'{speaker},{speakers}'
		if locationAMS:
			tdata['locationAMS'] = locationAMS
		matches = await main.search_scene_events_multi_speaker(show_key, speakers=all_speakers, location=locationAMS)
		tdata['episode_matches'] = matches['matches']
		tdata['episode_match_count'] = matches['episode_count']
		tdata['scene_match_count'] = matches['scene_count']
		tdata['scene_event_match_count'] = matches['scene_event_count']

	return templates.TemplateResponse('character.html', {'request': request, 'tdata': tdata})


# @web_app.get("/web/character_search/{show_key}/", response_class=HTMLResponse)
# async def character_search_page(request: Request, show_key: ShowKey, qt: str = None):
# 	tdata = {}

# 	tdata['header'] = 'character'
# 	tdata['show_key'] = show_key.value
# 	if not qt:
# 		tdata['qt'] = ''

# 	else:
# 		tdata['qt'] = qt
# 		# TODO
	
# 	return templates.TemplateResponse('characterSearch.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/character_listing/{show_key}/", response_class=HTMLResponse)
async def character_listing_page(request: Request, show_key: ShowKey, qt: str = None):
	tdata = {}

	tdata['header'] = 'character'
	tdata['show_key'] = show_key.value
	
	speaker_counts = await main.composite_speaker_aggs(show_key)
	tdata['speaker_stats'] = speaker_counts['speaker_agg_composite']

	tdata['speaker_matches'] = []
	if qt:
		tdata['qt'] = qt
		qt = qt.lower()
		# TODO I wish speaker_agg_composite were a dict instead of a list
		for sc in tdata['speaker_counts']:
			if qt in sc['speaker'].lower():
				tdata['speaker_matches'].append(sc)
	
	return templates.TemplateResponse('characterListing.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/graph/{show_key}")
async def show_page(request: Request, show_key: ShowKey, background_tasks: BackgroundTasks, num_clusters: int = 0):
	if not num_clusters:
		num_clusters = 4

	vector_field = 'openai_ada002_embeddings'
    # fetch all model/vendor embeddings for show 
	s = esqb.fetch_all_embeddings(show_key.value, vector_field)
	doc_embeddings = esrt.return_all_embeddings(s, vector_field)
    
    # cluster content
	doc_clusters, doc_clusters_df, embeddings_matrix = ef.cluster_docs(doc_embeddings, num_clusters)
    # doc_clusters_df.set_index('doc_id').T.to_dict('list')
	# doc_clusters_df.to_dict('dict')
	
	# clusters = main.cluster_content(show_key, num_clusters)
	img_buf = dz.generate_graph(doc_clusters_df, embeddings_matrix, num_clusters)
	background_tasks.add_task(img_buf.close)
	headers = {'Content-Disposition': 'inline; filename="out.png"'}
	return Response(img_buf.getvalue(), headers=headers, media_type='image/png')
