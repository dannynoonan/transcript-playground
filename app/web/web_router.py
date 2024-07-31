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
	list_seasons_response = esr.list_seasons(show_key)
	tdata['all_seasons'] = list_seasons_response['seasons']

	tdata['season_count'] = len(tdata['all_seasons'])
	series_locations_response = esr.agg_scenes_by_location(show_key)
	tdata['location_count'] = series_locations_response['location_count']
	series_speakers_response = esr.agg_scene_events_by_speaker(show_key)
	tdata['line_count'] = series_speakers_response['scene_events_by_speaker']['_ALL_']
	series_speaker_scene_counts_response = esr.agg_scenes_by_speaker(show_key)
	tdata['scene_count'] = series_speaker_scene_counts_response['scenes_by_speaker']['_ALL_']
	series_speaker_episode_counts_response = esr.agg_episodes_by_speaker(show_key)
	tdata['speaker_count'] = series_speaker_episode_counts_response['speaker_count']	
	series_speaker_word_counts_response = esr.agg_dialog_word_counts(show_key)
	tdata['word_count'] = int(series_speaker_word_counts_response['dialog_word_counts']['_ALL_'])

	location_counts = esr.composite_location_aggs(show_key)
	tdata['location_counts'] = location_counts['location_agg_composite']

	indexed_speakers_response = esr.fetch_indexed_speakers(show_key, extra_fields='topics_mbti')
	tdata['indexed_speakers'] = indexed_speakers_response['speakers']

	keywords = esr.keywords_by_corpus(show_key, exclude_speakers=True)
	tdata['keywords'] = keywords['keywords']

	episodes_by_season = esr.list_simple_episodes_by_season(show_key)
	tdata['episodes_by_season'] = episodes_by_season['episodes_by_season']

	tdata['episode_count'] = 0
	first_episode_in_series = None
	last_episode_in_series = None
	stats_by_season = {}
	for season in tdata['episodes_by_season'].keys():
		season_episode_count = len(tdata['episodes_by_season'][season])
		tdata['episode_count'] += len(tdata['episodes_by_season'][season])
		stats = {}
		season_locations = esr.agg_scenes_by_location(show_key, season=season)
		stats['location_count'] = season_locations['location_count']
		stats['location_counts'] = utils.truncate_dict(season_locations['scenes_by_location'], season_episode_count, start_index=1)
		season_speakers = esr.agg_scene_events_by_speaker(show_key, season=season)
		stats['line_count'] = season_speakers['scene_events_by_speaker']['_ALL_']
		stats['speaker_line_counts'] = utils.truncate_dict(season_speakers['scene_events_by_speaker'], season_episode_count, start_index=1)
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
		if not first_episode_in_series:
			first_episode_in_series = tdata['episodes_by_season'][season][0]
		last_episode_in_series = tdata['episodes_by_season'][season][-1]
		stats_by_season[season] = stats
	tdata['stats_by_season'] = stats_by_season
	tdata['air_date_range'] = f"{first_episode_in_series['air_date'][:10]} - {last_episode_in_series['air_date'][:10]}"

	return templates.TemplateResponse("show.html", {"request": request, 'tdata': tdata})


@web_app.get("/web/season/{show_key}/{season}", response_class=HTMLResponse, tags=['Web'])
async def season_page(request: Request, show_key: ShowKey, season: str):
	tdata = {}

	tdata['header'] = 'season'
	tdata['show_key'] = show_key.value
	tdata['season'] = season
	list_seasons_response = esr.list_seasons(show_key)
	all_seasons = list_seasons_response['seasons']
	tdata['all_seasons'] = all_seasons

	tdata['prev_season'] = None
	tdata['next_season'] = None
	season_index = all_seasons.index(int(season))
	if season_index > 0:
		tdata['prev_season'] = all_seasons[season_index-1]
	if season_index < len(all_seasons)-1:
		tdata['next_season'] = all_seasons[season_index+1]

	locations_by_scene = esr.agg_scenes_by_location(show_key, season=season)
	tdata['locations_by_scene'] = locations_by_scene['scenes_by_location']

	location_counts = esr.composite_location_aggs(show_key, season=season)
	tdata['location_counts'] = location_counts['location_agg_composite']

	speaker_seasons_response = esr.fetch_speakers_for_season(show_key, season)
	speaker_seasons = speaker_seasons_response['speaker_seasons']

	speaker_season_topics_response = esr.fetch_speaker_season_topics(show_key, 'meyersBriggsKiersey', season=season, level='child')
	speaker_season_topics = speaker_season_topics_response['speaker_season_topics']
	for speaker_season in speaker_seasons:
		speaker = speaker_season['speaker']
		if speaker in speaker_season_topics:
			speaker_season['topics_mbti'] = speaker_season_topics[speaker]
	tdata['speaker_seasons'] = speaker_seasons

	keywords = esr.keywords_by_corpus(show_key, season=season, exclude_speakers=True)
	tdata['keywords'] = keywords['keywords']

	episodes_response = esr.fetch_simple_episodes(show_key, season=season)
	episodes = episodes_response['episodes']
	tdata['episode_count'] = len(episodes)
	tdata['episodes'] = episodes

	locations = esr.agg_scenes_by_location(show_key, season=season)
	tdata['location_count'] = locations['location_count']

	# TODO this is what gets replaced by fetch_indexed_speakers, right?
	speaker_line_aggs_response = esr.agg_scene_events_by_speaker(show_key, season=season)
	tdata['line_count'] = speaker_line_aggs_response['scene_events_by_speaker']['_ALL_']

	speaker_scene_aggs_response = esr.agg_scenes_by_speaker(show_key, season=season)
	tdata['scene_count'] = speaker_scene_aggs_response['scenes_by_speaker']['_ALL_']

	speaker_episode_aggs_response = esr.agg_episodes_by_speaker(show_key, season=season)
	tdata['speaker_count'] = speaker_episode_aggs_response['speaker_count']	
	
	speaker_wc_aggs_response = esr.agg_dialog_word_counts(show_key, season=season)
	tdata['word_count'] = int(speaker_wc_aggs_response['dialog_word_counts']['_ALL_'])
	
	# generate air_date_range
	first_episode_in_season = episodes[0]
	last_episode_in_season = episodes[-1]
	tdata['air_date_range'] = f"{first_episode_in_season['air_date'][:10]} - {last_episode_in_season['air_date'][:10]}"

	return templates.TemplateResponse("season.html", {"request": request, 'tdata': tdata})


@web_app.get("/web/episode/{show_key}/{episode_key}", response_class=HTMLResponse, tags=['Web'])
def episode_page(request: Request, show_key: ShowKey, episode_key: str, search_type: str = None, qt: str = None, dialog: str = None, 
				 speaker: str = None, location: str = None, speakers: str = None, locationAMS: str = None):
	tdata = {}

	tdata['header'] = 'episode'
	tdata['show_key'] = show_key.value
	tdata['episode_key'] = episode_key
	list_seasons_response = esr.list_seasons(show_key)
	tdata['all_seasons'] = list_seasons_response['seasons']

	episode = esr.fetch_episode(show_key, episode_key)
	tdata['episode'] = episode['es_episode']
	
	scenes_by_location_response = esr.agg_scenes_by_location(show_key, episode_key=episode_key)
	tdata['locations_by_scene'] = scenes_by_location_response['scenes_by_location']
	tdata['scene_count'] = tdata['locations_by_scene']['_ALL_']
	del tdata['locations_by_scene']['_ALL_']

	scene_events_by_speaker_response = esr.agg_scene_events_by_speaker(show_key, episode_key=episode_key)
	tdata['line_count'] = scene_events_by_speaker_response['scene_events_by_speaker']['_ALL_']

	dialog_word_counts_response = esr.agg_dialog_word_counts(show_key, episode_key=episode_key)
	tdata['episode_word_counts'] = dialog_word_counts_response['dialog_word_counts']
	tdata['word_count'] = round(tdata['episode_word_counts']['_ALL_'])
	del tdata['episode_word_counts']['_ALL_']

	speaker_episodes_response = esr.fetch_speakers_for_episode(show_key, episode_key, extra_fields='topics_mbti')
	tdata['speaker_episodes'] = speaker_episodes_response['speaker_episodes']
	
	keywords = esr.keywords_by_episode(show_key, episode_key, exclude_speakers=True)
	tdata['keywords'] = keywords['keywords']
	
	mlt_tfidf = esr.more_like_this(show_key, episode_key)
	tdata['mlt_tfidf'] = mlt_tfidf['matches']

	mlt_embeddings = esr.episode_mlt_vector_search(show_key, episode_key)
	tdata['mlt_embeddings'] = mlt_embeddings['matches'][:30]

	tdata['topics_by_grouping'] = {}
	for topic_grouping in EPISODE_TOPIC_GROUPINGS:
		sort_by = 'score'
		if topic_grouping in ['universalGenres', 'focusedGpt35_TNG']:
			sort_by = 'tfidf_score'
		episode_topics_response = esr.fetch_episode_topics(show_key, episode_key, topic_grouping, limit=50, sort_by=sort_by)
		tdata['topics_by_grouping'][topic_grouping] = episode_topics_response['episode_topics']

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
		matches = esr.search_scene_events_multi_speaker(show_key, speakers, episode_key=episode_key, location=locationAMS, intersection=True)
		if matches and matches['matches']:
			tdata['episode_match'] = matches['matches'][0]
			tdata['scene_match_count'] = matches['scene_count']
			tdata['scene_event_match_count'] = matches['scene_event_count']

	return templates.TemplateResponse('episode.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/episode_search/{show_key}", response_class=HTMLResponse, tags=['Web'])
def episode_search_page(request: Request, show_key: ShowKey, search_type: str = None, season: str = None, qt: str = None, dialog: str = None, 
						speaker: str = None, location: str = None, qtSemantic: str = None, model_vendor: str = None, model_version: str = None, 
						speakers: str = None, locationAMS: str = None):
	tdata = {}

	tdata['header'] = 'search'
	tdata['show_key'] = show_key.value
	if not search_type:
		tdata['search_type'] = ''
	else:
		tdata['search_type'] = search_type
	tdata['season'] = season
	list_seasons_response = esr.list_seasons(show_key)
	tdata['all_seasons'] = list_seasons_response['seasons']

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
		matches = esr.episode_vector_search(show_key, qt=qtSemantic, model_vendor=model_vendor, model_version=model_version)
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
		matches = esr.search_scene_events_multi_speaker(show_key, speakers, season=season, location=locationAMS, intersection=True)
		tdata['episode_matches'] = matches['matches']
		tdata['episode_match_count'] = matches['episode_count']
		tdata['scene_match_count'] = matches['scene_count']
		tdata['scene_event_match_count'] = matches['scene_event_count']

	else:
		print(f'unsupported search_type={search_type}')

	return templates.TemplateResponse('episodeSearch.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/character/{show_key}/{speaker}", response_class=HTMLResponse, tags=['Web'])
def character_page(request: Request, show_key: ShowKey, speaker: str, search_type: str = None, season: str = None, dialog: str = None, 
				   location: str = None, speakers: str = None, locationAMS: str = None):
	tdata = {}

	tdata['header'] = 'character'
	tdata['show_key'] = show_key.value
	tdata['speaker'] = speaker
	list_seasons_response = esr.list_seasons(show_key)
	tdata['all_seasons'] = list_seasons_response['seasons']

	speaker_es_response = esr.fetch_speaker(show_key, speaker, include_seasons=True, include_episodes=True)
	if 'speaker' in speaker_es_response:
		es_speaker = speaker_es_response['speaker']
		if 'episodes' in es_speaker:
			tdata['episodes'] = es_speaker['episodes']
		if 'seasons' in es_speaker:
			tdata['seasons'] = es_speaker['seasons']
		tdata['season_count'] = es_speaker['season_count']
		tdata['episode_count'] = es_speaker['episode_count']
		tdata['scene_count'] = es_speaker['scene_count']
		tdata['line_count'] = es_speaker['line_count']
		tdata['word_count'] = es_speaker['word_count']
		if 'actor_names' in es_speaker:
			tdata['actor_names'] = es_speaker['actor_names']
		if 'alt_names' in es_speaker:
			tdata['alt_names'] = [name for name in es_speaker['alt_names'] if name.upper() != speaker]

		# inject full spectrum of speaker topics
		tdata['child_topics_by_grouping'] = {}
		tdata['parent_topics_by_grouping'] = {}
		for topic_grouping in SPEAKER_TOPIC_GROUPINGS:
			child_topics_response = esr.fetch_speaker_topics(speaker, show_key, topic_grouping, level='child')
			tdata['child_topics_by_grouping'][topic_grouping] = child_topics_response['speaker_topics']
			parent_topics_response = esr.fetch_speaker_topics(speaker, show_key, topic_grouping, level='parent')
			tdata['parent_topics_by_grouping'][topic_grouping] = parent_topics_response['speaker_topics']

	# TODO legacy, would love to remove this
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

	speaker_mlt_response = esr.speaker_mlt_vector_search(show_key, speaker)
	tdata['speaker_mlt_aggs'] = speaker_mlt_response['all_speaker_matches']
	# TODO struggling with preserving season sorting
	tdata['speaker_mlt_series_matches'] = speaker_mlt_response['matches_by_speaker_series_embedding']
	tdata['speaker_mlt_season_matches'] = speaker_mlt_response['matches_by_speaker_season_embedding']
	tdata['speaker_mlt_episode_matches'] = speaker_mlt_response['matches_by_speaker_episode_embedding']

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
		matches = esr.search_scene_events_multi_speaker(show_key, all_speakers, location=locationAMS, intersection=True)
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
def character_listing_page(request: Request, show_key: ShowKey, qt: str = None):
	tdata = {}

	tdata['header'] = 'character'
	tdata['show_key'] = show_key.value
	list_seasons_response = esr.list_seasons(show_key)
	tdata['all_seasons'] = list_seasons_response['seasons']

	indexed_speakers_response = esr.fetch_indexed_speakers(show_key, extra_fields='topics_mbti')
	indexed_speakers = indexed_speakers_response['speakers']
	tdata['indexed_speakers'] = indexed_speakers

	# TODO well THIS is inefficient...
	indexed_speaker_keys = [s['speaker'] for s in indexed_speakers]
	speaker_aggs_response = esr.composite_speaker_aggs(show_key)
	speaker_aggs = speaker_aggs_response['speaker_agg_composite']
	tdata['non_indexed_speakers'] = [s for s in speaker_aggs if s['speaker'] not in indexed_speaker_keys]

	tdata['speaker_matches'] = []
	if qt:
		tdata['qt'] = qt
		speaker_search_response = esr.search_speakers(qt, show_key=show_key)
		tdata['speaker_matches'] = speaker_search_response['speaker_matches']
	
	return templates.TemplateResponse('characterListing.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/topic_listing/{show_key}", response_class=HTMLResponse, tags=['Web'])
def topic_listing_page(request: Request, show_key: ShowKey, selected_topic_grouping: str = None):
	tdata = {}

	if not selected_topic_grouping:
		selected_topic_grouping = EPISODE_TOPIC_GROUPINGS[0]

	tdata['header'] = 'topic'
	tdata['show_key'] = show_key.value
	tdata['selected_topic_grouping'] = selected_topic_grouping
	list_seasons_response = esr.list_seasons(show_key)
	tdata['all_seasons'] = list_seasons_response['seasons']

	tdata['topic_groupings'] = {}

	# TODO will probably split up episode and speaker topics, for now keeping together
	topic_groupings = EPISODE_TOPIC_GROUPINGS + SPEAKER_TOPIC_GROUPINGS

	for tg in topic_groupings:
		# if tg.startswith('focused'):
		# 	tg = f'{tg}_{show_key.value}'
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
def topic_page(request: Request, show_key: ShowKey, topic_grouping: str, topic_key: str):
	tdata = {}

	tdata['header'] = 'topic'
	tdata['show_key'] = show_key.value
	tdata['topic_grouping'] = topic_grouping
	tdata['topic_key'] = topic_key
	list_seasons_response = esr.list_seasons(show_key)
	tdata['all_seasons'] = list_seasons_response['seasons']

	tdata['episodes'] = []
	tdata['speakers'] = []

	topic_response = esr.fetch_topic(topic_grouping, topic_key)
	tdata['topic'] = topic_response['topic']
	tdata['topic']['breadcrumb'] = tdata['topic']['topic_key']
	if tdata['topic']['parent_key']:
		tdata['topic']['breadcrumb'] = f"{tdata['topic']['parent_key']} > {tdata['topic']['breadcrumb']}"

	if topic_grouping in EPISODE_TOPIC_GROUPINGS:
		episode_topic_response = esr.find_episodes_by_topic(show_key, topic_grouping, topic_key, sort_by='score')
		tdata['episode_topics'] = episode_topic_response['episode_topics']
		episode_topic_response_raw_sort = esr.find_episodes_by_topic(show_key, topic_grouping, topic_key, sort_by='raw_score')
		tdata['episode_topics_raw_sort'] = episode_topic_response_raw_sort['episode_topics']
			
	elif topic_grouping in SPEAKER_TOPIC_GROUPINGS:
		speaker_topic_response = esr.find_speakers_by_topic(topic_grouping, topic_key, show_key=show_key, min_word_count=3000)
		tdata['speaker_topics'] = speaker_topic_response['speaker_topics']

		speaker_season_topic_response = esr.find_speaker_seasons_by_topic(topic_grouping, topic_key, show_key, min_word_count=2000)
		tdata['speaker_season_topics'] = speaker_season_topic_response['speaker_season_topics']

		speaker_episode_topic_response = esr.find_speaker_episodes_by_topic(topic_grouping, topic_key, show_key, min_word_count=1000)
		tdata['speaker_episode_topics'] = speaker_episode_topic_response['speaker_episode_topics']
	
	return templates.TemplateResponse('topic.html', {'request': request, 'tdata': tdata})


@web_app.get("/web/graph/{show_key}", response_class=HTMLResponse, tags=['Web'])
def graph_page(request: Request, show_key: ShowKey, background_tasks: BackgroundTasks, num_clusters: int = 0):
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
