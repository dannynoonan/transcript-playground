from fastapi import APIRouter, Request, Response, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from operator import itemgetter

import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.es.es_read_router as esr
import app.nlp.embeddings_factory as ef
from app.show_metadata import ShowKey, EPISODE_TOPIC_GROUPINGS, SPEAKER_TOPIC_GROUPINGS
import app.utils as utils
import app.web.fig_builder as fb


templates = Jinja2Templates(directory="app/templates")
web_app = APIRouter()
# web_app.mount('/static', StaticFiles(directory='static', html=True), name='static')


@web_app.get("/web/show/{show_key}", response_class=HTMLResponse, tags=['Web'])
async def show_page(request: Request, show_key: ShowKey):
	tdata = {}

	tdata['header'] = 'show'
	tdata['show_key'] = show_key.value

	# locations_by_scene = await esr.agg_scenes_by_location(show_key)
	# tdata['locations_by_scene'] = locations_by_scene['scenes_by_location']

	location_counts = esr.composite_location_aggs(show_key)
	tdata['location_counts'] = location_counts['location_agg_composite']

	speaker_counts = esr.composite_speaker_aggs(show_key)
	tdata['speaker_counts'] = speaker_counts['speaker_agg_composite']

	keywords = esr.keywords_by_corpus(show_key, exclude_speakers=True)
	tdata['keywords'] = keywords['keywords']

	episodes_by_season = esr.list_simple_episodes_by_season(show_key)
	tdata['episodes_by_season'] = episodes_by_season['episodes_by_season']

	stats_by_season = {}
	for season in tdata['episodes_by_season'].keys():
		season_episode_count = len(tdata['episodes_by_season'][season])
		stats = {}
		season_locations = esr.agg_scenes_by_location(show_key, season=season)
		stats['location_count'] = season_locations['location_count']
		stats['location_counts'] = utils.truncate_dict(season_locations['scenes_by_location'], season_episode_count, 1)
		season_speakers = esr.agg_scene_events_by_speaker(show_key, season=season)
		stats['line_count'] = season_speakers['scene_events_by_speaker']['_ALL_']
		stats['speaker_line_counts'] = utils.truncate_dict(season_speakers['scene_events_by_speaker'], season_episode_count, 1)
		season_speaker_scene_counts = esr.agg_scenes_by_speaker(show_key, season=season)
		stats['scene_count'] = season_speaker_scene_counts['scenes_by_speaker']['_ALL_']
		season_speaker_episode_counts = esr.agg_episodes_by_speaker(show_key, season=season)
		stats['speaker_count'] = season_speaker_episode_counts['speaker_count']	
		season_speaker_word_counts = esr.agg_dialog_word_counts(show_key, season=season)
		stats['word_count'] = int(season_speaker_word_counts['dialog_word_counts']['_ALL_'])
		# generate air_date_range
		first_episode_in_season = tdata['episodes_by_season'][season][0]
		last_episode_in_season = tdata['episodes_by_season'][season][-1]
		stats['air_date_range'] = f"{first_episode_in_season['air_date'][:10]} - {last_episode_in_season['air_date'][:10]}"
		stats_by_season[season] = stats
	tdata['stats_by_season'] = stats_by_season

	return templates.TemplateResponse("show.html", {"request": request, 'tdata': tdata})


@web_app.get("/web/episode/{show_key}/{episode_key}", response_class=HTMLResponse, tags=['Web'])
def episode_page(request: Request, show_key: ShowKey, episode_key: str, search_type: str = None, qt: str = None,
					   dialog: str = None, speaker: str = None, location: str = None, speakers: str = None, locationAMS: str = None):
	tdata = {}

	tdata['header'] = 'episode'
	tdata['show_key'] = show_key.value
	tdata['episode_key'] = episode_key

	episode = esr.fetch_episode(show_key, episode_key)
	tdata['episode'] = episode['es_episode']
	
	locations_by_scene = esr.agg_scenes_by_location(show_key, episode_key=episode_key)
	tdata['locations_by_scene'] = locations_by_scene['scenes_by_location']

	episode_word_counts = esr.agg_dialog_word_counts(show_key, episode_key=episode_key)
	tdata['episode_word_counts'] = episode_word_counts['dialog_word_counts']

	speaker_counts = esr.composite_speaker_aggs(show_key, episode_key=episode_key)
	tdata['speaker_counts'] = speaker_counts['speaker_agg_composite']
	
	keywords = esr.keywords_by_episode(show_key, episode_key, exclude_speakers=True)
	tdata['keywords'] = keywords['keywords']
	
	mlt_tfidf = esr.more_like_this(show_key, episode_key)
	tdata['mlt_tfidf'] = mlt_tfidf['matches']

	mlt_embeddings = esr.mlt_vector_search(show_key, episode_key)
	tdata['mlt_embeddings'] = mlt_embeddings['matches'][:30]

	tdata['topic_embeddings'] = {}
	for tg in EPISODE_TOPIC_GROUPINGS:
		if tg.startswith('focused'):
			tg = f'{tg}_{show_key.value}'
		topic_embeddings = esr.episode_topic_vector_search(show_key, episode_key, tg, 'openai', 'ada002')
		tdata['topic_embeddings'][tg] = topic_embeddings['topics'][:30]

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
		matches = esr.search(show_key, episode_key=episode_key, qt=qt)
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
			matches = esr.search_scenes(show_key, episode_key=episode_key, location=location)
		else:
			matches = esr.search_scene_events(show_key, episode_key=episode_key, speaker=speaker, dialog=dialog, location=location)
			tdata['scene_event_match_count'] = matches['scene_event_count']
		if matches['matches']:
			tdata['episode_match'] = matches['matches'][0]
			tdata['scene_match_count'] = matches['scene_count']
		else:
			tdata['episode_match'] = ''
			tdata['scene_match_count'] = 0
		
	elif search_type == 'advanced_multi_speaker':
		if speakers:
			tdata['speakers'] = speakers
		if locationAMS:
			tdata['locationAMS'] = locationAMS
		matches = esr.search_scene_events_multi_speaker(show_key, episode_key=episode_key, speakers=speakers, location=locationAMS)
		if matches and matches['matches']:
			tdata['episode_match'] = matches['matches'][0]
			tdata['scene_match_count'] = matches['scene_count']
			tdata['scene_event_match_count'] = matches['scene_event_count']

	return templates.TemplateResponse('episode.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/episode_search/{show_key}", response_class=HTMLResponse, tags=['Web'])
def episode_search_page(request: Request, show_key: ShowKey, search_type: str = None, season: str = None, qt: str = None, 
							  dialog: str = None, speaker: str = None, location: str = None, qtSemantic: str = None, model_vendor: str = None, 
							  model_version: str = None, speakers: str = None, locationAMS: str = None):
	tdata = {}

	tdata['header'] = 'search'
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
		matches = esr.search(show_key, season=season, qt=qt)
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
			matches = esr.search_scenes(show_key, season=season, location=location)
		else:
			matches = esr.search_scene_events(show_key, season=season, speaker=speaker, dialog=dialog, location=location)
			tdata['scene_event_match_count'] = matches['scene_event_count']
		tdata['episode_matches'] = matches['matches']
		tdata['episode_match_count'] = matches['episode_count']
		tdata['scene_match_count'] = matches['scene_count']

	elif search_type == 'semantic':
		tdata['qtSemantic'] = qtSemantic
		if not model_vendor:
			model_vendor = 'openai'
		if not model_version:
			model_version = 'ada002'
		tdata['model_vendor'] = model_vendor
		tdata['model_version'] = model_version
		matches = esr.vector_search(show_key, qt=qtSemantic, model_vendor=model_vendor, model_version=model_version)
		print(f'############ matches={matches}')
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
		matches = esr.search_scene_events_multi_speaker(show_key, season=season, speakers=speakers, location=locationAMS)
		tdata['episode_matches'] = matches['matches']
		tdata['episode_match_count'] = matches['episode_count']
		tdata['scene_match_count'] = matches['scene_count']
		tdata['scene_event_match_count'] = matches['scene_event_count']

	else:
		print(f'unsupported search_type={search_type}')

	return templates.TemplateResponse('episodeSearch.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/character/{show_key}/{speaker}", response_class=HTMLResponse, tags=['Web'])
def character_page(request: Request, show_key: ShowKey, speaker: str, search_type: str = None, season: str = None,
						 dialog: str = None, location: str = None, speakers: str = None, locationAMS: str = None):
	tdata = {}

	tdata['header'] = 'character'
	tdata['show_key'] = show_key.value
	tdata['speaker'] = speaker

	response = esr.fetch_speaker(show_key, speaker, include_seasons=True, include_episodes=True)
	if 'speaker' in response:
		es_speaker = response['speaker']
		if 'episodes' in es_speaker:
			tdata['episodes'] = es_speaker['episodes']
		else:
			# TODO if this happens something is wrong
			tdata['episodes'] = []
		if 'seasons' in es_speaker:
			tdata['seasons'] = es_speaker['seasons']
		else:
			# TODO if this happens something is wrong
			tdata['seasons'] = []
		tdata['season_count'] = es_speaker['season_count']
		tdata['episode_count'] = es_speaker['episode_count']
		tdata['scene_count'] = es_speaker['scene_count']
		tdata['line_count'] = es_speaker['line_count']
		tdata['word_count'] = es_speaker['word_count']
		if 'actor_names' in es_speaker:
			tdata['actor_names'] = es_speaker['actor_names']
		else:
			tdata['actor_names'] = []
		if 'alt_names' in es_speaker:
			tdata['alt_names'] = [name for name in es_speaker['alt_names'] if name.upper() != speaker]
		else:
			tdata['alt_names'] = []
		tdata['parent_topics'] = es_speaker['parent_topics']
		tdata['child_topics'] = es_speaker['child_topics']
		print(f"tdata['child_topics']={tdata['child_topics']}")
	else:
		episode_matches = esr.search_scene_events(show_key, speaker=speaker)
		for m in episode_matches['matches']:
			m['line_count'] = m['scene_event_count'] # understandable but sloppy naming inconsistency
		tdata['episodes'] = episode_matches['matches']
		tdata['episode_count'] = episode_matches['episode_count']
		tdata['scene_count'] = episode_matches['scene_count']
		tdata['line_count'] = episode_matches['scene_event_count']
		word_count = esr.agg_dialog_word_counts(show_key, speaker=speaker)
		tdata['word_count'] = int(word_count['dialog_word_counts'][speaker])
		tdata['seasons'] = []
		tdata['actor_names'] = []
		tdata['alt_names'] = []
		tdata['parent_topics'] = {}
		tdata['child_topics'] = {}

	locations_counts = esr.agg_scenes_by_location(show_key, speaker=speaker)
	tdata['location_counts'] = locations_counts['scenes_by_location']

	co_occ_speakers_by_episode = esr.agg_episodes_by_speaker(show_key, other_speaker=speaker)
	co_occ_speakers_by_scene = esr.agg_scenes_by_speaker(show_key, other_speaker=speaker)
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
		matches = esr.search_scene_events(show_key, speaker=speaker, dialog=dialog, location=location)
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
		matches = esr.search_scene_events_multi_speaker(show_key, speakers=all_speakers, location=locationAMS)
		tdata['episode_matches'] = matches['matches']
		tdata['episode_match_count'] = matches['episode_count']
		tdata['scene_match_count'] = matches['scene_count']
		tdata['scene_event_match_count'] = matches['scene_event_count']

	return templates.TemplateResponse('character.html', {'request': request, 'tdata': tdata})


# @web_app.get("/web/character_search/{show_key}/", response_class=HTMLResponse, tags=['Web'])
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


@web_app.get("/web/character_listing/{show_key}/", response_class=HTMLResponse, tags=['Web'])
async def character_listing_page(request: Request, show_key: ShowKey, qt: str = None):
	tdata = {}

	tdata['header'] = 'character'
	tdata['show_key'] = show_key.value
	
	speaker_counts = esr.composite_speaker_aggs(show_key)
	tdata['speaker_stats'] = speaker_counts['speaker_agg_composite']

	return_fields = ['speaker', 'season_count', 'episode_count', 'scene_count', 'line_count', 'word_count', 'parent_topics', 'child_topics']
	s = esqb.fetch_indexed_speakers(show_key.value, return_fields=return_fields)
	tdata['speakers'] = esrt.return_speakers(s)

	tdata['speaker_matches'] = []
	if qt:
		tdata['qt'] = qt
		qt = qt.lower()
		# TODO I wish speaker_agg_composite were a dict instead of a list
		for speaker in tdata['speaker_stats']:
			if qt in speaker['speaker'].lower():
				tdata['speaker_matches'].append(speaker)
	
	return templates.TemplateResponse('characterListing.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/topic_listing/{show_key}", response_class=HTMLResponse, tags=['Web'])
async def topic_listing_page(request: Request, show_key: ShowKey, selected_topic_grouping: str = None):
	tdata = {}

	if not selected_topic_grouping:
		selected_topic_grouping = EPISODE_TOPIC_GROUPINGS[0]

	tdata['header'] = 'topic'
	tdata['show_key'] = show_key.value
	tdata['selected_topic_grouping'] = selected_topic_grouping
	tdata['topic_groupings'] = {}

	# TODO will probably split up episode and speaker topics, for now keeping together
	topic_groupings = EPISODE_TOPIC_GROUPINGS + SPEAKER_TOPIC_GROUPINGS

	for tg in topic_groupings:
		if tg.startswith('focused'):
			tg = f'{tg}_{show_key.value}'
		response = esr.fetch_topic_grouping(tg)
		if 'topics' not in response:
			continue
		# sort by combination of parent and child topic_keys
		topics = response['topics']
		for t in topics:
			t['breadcrumb'] = t['topic_key']
			if t['parent_key']:
				t['breadcrumb'] = f"{t['parent_key']} > {t['breadcrumb']}"
		breadcrumb_sorted_topics = sorted(topics, key=itemgetter('breadcrumb'))
		tdata['topic_groupings'][tg] = breadcrumb_sorted_topics
	
	return templates.TemplateResponse('topicListing.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/topic/{show_key}/{topic_grouping}/{topic_key}", response_class=HTMLResponse, tags=['Web'])
async def topic_listing_page(request: Request, show_key: ShowKey, topic_grouping: str, topic_key: str):
	tdata = {}

	tdata['header'] = 'topic'
	tdata['show_key'] = show_key.value
	tdata['topic_grouping'] = topic_grouping
	tdata['topic_key'] = topic_key
	tdata['episodes'] = []
	tdata['speakers'] = []

	# TODO groan this is gross, I think it was temporary but I feel it getting baked in...
	focused_topic_grouping = topic_grouping
	if topic_grouping.startswith('focused'):
		focused_topic_grouping = f'{topic_grouping}_{show_key.value}'

	topic_response = esr.fetch_topic(focused_topic_grouping, topic_key)
	tdata['topic'] = topic_response['topic']
	tdata['topic']['breadcrumb'] = tdata['topic']['topic_key']
	if tdata['topic']['parent_key']:
		tdata['topic']['breadcrumb'] = f"{tdata['topic']['parent_key']} > {tdata['topic']['breadcrumb']}"

	if topic_grouping in EPISODE_TOPIC_GROUPINGS:
		vector_search_response = esr.topic_episode_vector_search(focused_topic_grouping, topic_key, show_key)
		if 'episodes' in vector_search_response:
			episodes = vector_search_response['episodes']
			if len(episodes) > 30:
				episodes = episodes[:30]
			tdata['episodes'] = episodes
			
	elif topic_grouping in SPEAKER_TOPIC_GROUPINGS:
		vector_search_response = esr.topic_speaker_vector_search(focused_topic_grouping, topic_key, show_key)
		if 'speakers' in vector_search_response:
			speakers = vector_search_response['speakers']
			if len(speakers) > 30:
				speakers = speakers[:30]
			tdata['speakers'] = speakers
	
	return templates.TemplateResponse('topic.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/graph/{show_key}", response_class=HTMLResponse, tags=['Web'])
async def show_page(request: Request, show_key: ShowKey, background_tasks: BackgroundTasks, num_clusters: int = 0):
	if not num_clusters:
		num_clusters = 4

	vector_field = 'openai_ada002_embeddings'
    # fetch all model/vendor embeddings for show 
	s = esqb.fetch_series_embeddings(show_key.value, vector_field)
	doc_embeddings = esrt.return_all_embeddings(s, vector_field)
    
    # cluster content
	doc_clusters_df = ef.cluster_docs(doc_embeddings, num_clusters)
    # doc_clusters_df.set_index('doc_id').T.to_dict('list')
	# doc_clusters_df.to_dict('dict')
	
	# clusters = esr.cluster_content(show_key, num_clusters)
	# img_buf = dz.generate_graph_matplotlib(doc_clusters_df, show_key.value, num_clusters, matrix=embeddings_matrix)
	img_buf = fb.build_cluster_scatter_matplotlib(doc_clusters_df, show_key.value, num_clusters)
	background_tasks.add_task(img_buf.close)
	headers = {'Content-Disposition': 'inline; filename="out.png"'}
	return Response(img_buf.getvalue(), headers=headers, media_type='image/png')
