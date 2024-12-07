from fastapi import APIRouter
import os
import pandas as pd

from app.config import settings
if settings.es_toggle == 'oss':
    from app.es.oss_model import EsEpisodeTranscript, EsEpisodeNarrativeSequence, EsSpeaker, EsSpeakerSeason, EsSpeakerEpisode, EsTopic
else:
    from app.es.es_model import EsEpisodeTranscript, EsEpisodeNarrativeSequence, EsSpeaker, EsSpeakerSeason, EsSpeakerEpisode, EsTopic

from app.auth import user_dependency, exit_if_unauthorized
import app.database.dao as dao
import app.data_service.field_flattener as fflat
from app.data_service.topic_aggregator import TopicAgg
import app.es.es_ingest_transformer as esit
import app.es.es_query_builder as esqb
import app.routers.es_read_router as esr
import app.nlp.embeddings_factory as ef
import app.nlp.narrative_extractor as ne
from app.nlp.nlp_metadata import ACTIVE_VENDOR_VERSIONS, TRANSFORMER_VENDOR_VERSIONS as TRF_MODELS
from app.show_metadata import ShowKey


esw_app = APIRouter(prefix='/esw', tags=['ES Writer'])


@esw_app.get("/index_episode/{show_key}/{episode_key}")
async def index_episode(show_key: ShowKey, episode_key: str, user: user_dependency):
    '''
    Fetch `Episode` entity from Postgres `transcript_db`, transform Tortoise object to ElasticSearch object, and write it to ElasticSearch index.
    '''
    exit_if_unauthorized(user, level='admin')
    
    # fetch episode, throw errors if not found
    episode = None
    try:
        episode = await dao.fetch_episode(show_key.value, episode_key, fetch_related=['scenes', 'events'])
    except Exception as e:
        return {"Error": f"Failure to fetch Episode having show_key={show_key} external_key={episode_key} (have run /load_episode_listing?): {e}"}
    if not episode:
        return {"Error": f"No Episode found having show_key={show_key} external_key={episode_key}. You may need to run /load_episode_listing first."}
    
    # transform to es-writable object and write to es
    try:
        es_episode = esit.to_es_episode(episode)

        # NOTE this is how I generated test data, and it was enough of a pain I don't want to delete it
        # import json
        # es_episode_dict = es_episode.to_dict()
        # es_episode_json = json.dumps(es_episode_dict, default=str, indent=4)
        # f = open(f"test_data/es/es_episode_{show_key}_{episode_key}.json", "w")
        # f.write(es_episode_json)
        # f.close()
    except Exception as e:
        return {"Error": f"Failure to transform Episode {show_key}:{episode_key} to es-writable version: {e}"}
    
    try:
        esqb.save_es_episode(es_episode)
    except Exception as e:
        return {"Error": f"Failure to save Episode {show_key}:{episode_key} to es: {e}"}

    return {"Success": f"Episode {show_key}_{episode_key} written to es index"}


@esw_app.get("/populate_focal_speakers/{show_key}")
def populate_focal_speakers(show_key: ShowKey, user: user_dependency, episode_key: str = None):
    '''
    For each episode, query ElasticSearch to count the number of lines spoken per character, then write the top 3 characters back to their own ElasticSearch field
    '''
    exit_if_unauthorized(user, level='admin')
    
    episodes_to_focal_speakers = esqb.populate_focal_speakers(show_key.value, episode_key=episode_key)
    return {"episodes_to_focal_speakers": episodes_to_focal_speakers}


@esw_app.get("/populate_focal_locations/{show_key}")
def populate_focal_locations(show_key: ShowKey, user: user_dependency, episode_key: str = None):
    '''
    For each episode, query ElasticSearch to count the number of scenes per location, then write the top 3 locations back to their own ElasticSearch field
    '''
    exit_if_unauthorized(user, level='admin')
    
    episodes_to_focal_locations = esqb.populate_focal_locations(show_key.value, episode_key=episode_key)
    return {"episodes_to_focal_locations": episodes_to_focal_locations}


@esw_app.get("/populate_episode_relations/{show_key}/{episode_key}/{model_vendor}/{model_version}")
def populate_episode_relations(show_key: ShowKey, episode_key: str, model_vendor: str, model_version: str, user: user_dependency, limit: int = 30):
    '''
    Query ElasticSearch for most similar episodes vis-a-vis a given model:vendor, then write the top X episode|score pairs to corresponding relations field
    '''
    exit_if_unauthorized(user, level='admin')
    
    if (model_vendor, model_version) not in ACTIVE_VENDOR_VERSIONS and (model_vendor, model_version) != ('es','mlt'):
        return {"error": f'invalid model_vendor:model_version combo {model_vendor}:{model_version}'}
 
    if (model_vendor, model_version) == ('es','mlt'):
        similar_episodes = esr.more_like_this(ShowKey(show_key), episode_key)
    else:
        similar_episodes = esr.episode_mlt_vector_search(ShowKey(show_key), episode_key, model_vendor=model_vendor, model_version=model_version)
    # only keep the episode keys and corresponding scores 
    # sim_eps = [f"{sim_ep['episode_key']}|{sim_ep['score']}" for sim_ep in similar_episodes['matches']]
    # sim_eps = [(sim_ep['episode_key'], sim_ep['score']) for sim_ep in similar_episodes['matches']]
    # sim_eps = {sim_ep['episode_key']:sim_ep['score'] for sim_ep in similar_episodes['matches']}

    episode_relations = {}
    doc_id = f'{show_key}_{episode_key}'
    episode_relations[doc_id] = similar_episodes
    
    episode_relations = esqb.populate_episode_relations(show_key.value, model_vendor, model_version, episode_relations, limit=limit)

    return {"episode_relations": episode_relations}


# @esw_app.get("/build_embeddings_model/{show_key}")
# def build_embeddings_model(show_key: ShowKey):
#     '''
#     Experimental endpoint: goes thru the motions of building a language model using Word2Vec, but limits training data to a single show's text corpus, resulting in a (uselessly) tiny model
#     '''
#     model_info = ef.build_embeddings_model(show_key.value)
#     return {"model_info": model_info}


@esw_app.get("/populate_episode_embeddings/{show_key}/{episode_key}/{model_vendor}/{model_version}")
def populate_episode_embeddings(show_key: ShowKey, episode_key: str, model_vendor: str, model_version: str, user: user_dependency):
    '''
    Generate vector embedding for episode using pre-trained Word2Vec and Transformer models (enumerated in `nlp/nlp_metadata.py`)
    '''
    exit_if_unauthorized(user, level='admin')
    
    es_episode = EsEpisodeTranscript.get(id=f'{show_key.value}_{episode_key}')
    try:
        embeddings = ef.generate_episode_embeddings(es_episode, model_vendor, model_version)
        es_episode[f'{model_vendor}_{model_version}_embeddings'] = embeddings
        esqb.save_es_episode(es_episode)
        return {"es_episode": es_episode}
    except Exception as e:
        return {f"Failed to populate {model_vendor}:{model_version} embeddings for episode {show_key.value}_{episode_key}": e}


@esw_app.get("/index_speaker/{show_key}/{speaker}")
def index_speaker(show_key: ShowKey, speaker: str, user: user_dependency):
    '''
    Combine aggregate episode and series count data and dialog text for a series speaker with metadata loaded from csv file into standalone speaker index
    In addition to speakers index, also populate season-level and episode-level speaker indices for speaker
    '''
    exit_if_unauthorized(user, level='admin')
    
    es_speaker = EsSpeaker(show_key=show_key.value, speaker=speaker, scene_count=0, line_count=0, word_count=0, lines=[], seasons_to_episode_keys={})
    es_speaker_seasons = {}
    es_speaker_episodes = {}

    response = esr.search_scene_events(show_key, speaker=speaker)
    if 'matches' not in response:
        return {"error": f"No scene_events found matching show_key={show_key.value} speaker={speaker}"}
    
    # TODO store cast source csv file/dataframe in memory, or operate on it more wholistically than this one-off per speaker flow
    file_path = f'./source/speakers/{show_key.value}_cast.csv'
    if os.path.isfile(file_path):
        print(f'Loading cast dataframe from file_path={file_path}')
        cast_df = pd.read_csv(file_path)
        cast_df = cast_df.fillna('')
        speaker_rows = cast_df.loc[cast_df['Key'] == speaker]
        if len(speaker_rows) > 0:
            if len(speaker_rows) > 1:
                print(f'Warning: multiple rows in file_path={file_path} matched speaker={speaker}, using data from first result in series')
            # pandas series are weird
            es_speaker.alt_names = speaker_rows['Speaker names'].values[0].split('|')
            es_speaker.actor_names = speaker_rows['Actor names'].values[0].split('|')
    
    for episode in response['matches']:
        season = str(episode['season'])
        episode_key = episode['episode_key']
        es_speaker_episode = EsSpeakerEpisode(show_key=show_key.value, speaker=speaker, episode_key=episode_key, title=episode['title'], 
                                              air_date=episode['air_date'], season=season, sequence_in_season=episode['sequence_in_season'], 
                                              agg_score=episode['agg_score'], scene_count=0, line_count=0, word_count=0, lines=[])
        print(f'init-ing es_speaker_episode={es_speaker_episode} with es_speaker_episode.episode_key={es_speaker_episode.episode_key}')
        es_speaker_episodes[episode_key] = es_speaker_episode
        if season in es_speaker_seasons:
            es_speaker_season = es_speaker_seasons[season]
            es_speaker.seasons_to_episode_keys[season].append(episode_key)
        else:
            es_speaker_season = EsSpeakerSeason(show_key=show_key.value, speaker=speaker, season=season, episode_count=0, scene_count=0, 
                                                line_count=0, word_count=0, lines=[])
            es_speaker_seasons[season] = es_speaker_season
            es_speaker.seasons_to_episode_keys[season] = [episode_key]
        for scene in episode['scenes']:
            es_speaker_episode.scene_count += 1
            for scene_event in scene['scene_events']:
                es_speaker_episode.line_count += 1
                es_speaker_episode.word_count += len(scene_event['dialog'].split(' '))
                es_speaker_episode.lines.append(scene_event['dialog'])

        # add episode scene/line/word data to aggregate season and overall data 
        es_speaker_season.episode_count += 1
        es_speaker_season.scene_count += es_speaker_episode.scene_count
        es_speaker_season.line_count += es_speaker_episode.line_count
        es_speaker_season.word_count += es_speaker_episode.word_count
        es_speaker_season.lines.extend(es_speaker_episode.lines)
        es_speaker.scene_count += es_speaker_episode.scene_count
        es_speaker.line_count += es_speaker_episode.line_count
        es_speaker.word_count += es_speaker_episode.word_count
        es_speaker.lines.extend(es_speaker_episode.lines)

    es_speaker.season_count = len(es_speaker_seasons)
    es_speaker.episode_count = len(es_speaker_episodes)
    # special handling of openai token counters using `tiktoken`  
    es_speaker.openai_word_count = ef.openai_token_counter(' '.join(es_speaker.lines), 'cl100k_base')
    for _, ess in es_speaker_seasons.items():
        ess.openai_word_count = ef.openai_token_counter(' '.join(ess.lines), 'cl100k_base')
    for _, ese in es_speaker_episodes.items():
        ese.openai_word_count = ef.openai_token_counter(' '.join(ese.lines), 'cl100k_base')
    
    # write to es
    try:    
        print(f'Writing es_speaker {speaker} show_key={show_key.value} to `speakers`, `speaker_seasons`, `speaker_episodes`, and `speaker_embeddings_unified` indexes')  
        esqb.save_es_speaker(es_speaker)
        for _, es_speaker_season in es_speaker_seasons.items():
            esqb.save_es_speaker_season(es_speaker_season)
        for _, es_speaker_episode in es_speaker_episodes.items():
            print(f'saving es_speaker_episode={es_speaker_episode} with es_speaker_episode.episode_key={es_speaker_episode.episode_key}')
            esqb.save_es_speaker_episode(es_speaker_episode)
    except Exception as e:
        return {"error": f"Failure indexing speaker lines and counts for speaker={speaker} show_key={show_key.value}: {e}"}

    return {"speaker": speaker, "season_count": len(es_speaker_seasons), "episode_count": len(es_speaker_episodes)}


@esw_app.get("/index_topic_grouping/{topic_grouping}")
def index_topic_grouping(topic_grouping: str, user: user_dependency):
    '''
    Load set of Topics from csv file into es `topics` index.
    '''
    exit_if_unauthorized(user, level='admin')
    
    file_path = f'./source/topics/{topic_grouping}.csv'
    if os.path.isfile(file_path):
        print(f'Loading topic_grouping dataframe from file_path={file_path}')
        topics_df = pd.read_csv(file_path)
        topics_df = topics_df.fillna('')
        # for child categories, adopt parent category descriptions into parent_description field, for use in generating embeddings with fuller context
        parent_keys = topics_df['parent_key'].unique()
        print(f'parent_keys={parent_keys}')
        for parent_key in parent_keys:
            if parent_key == '':
                continue
            parent_desc_series = topics_df[(topics_df['topic_key'] == parent_key) & (topics_df['parent_key'] == '')]['description']
            print(f'parent_key={parent_key} parent_desc_series={parent_desc_series}')
            parent_desc = parent_desc_series.values[0] # NOTE feels like there should be a cleaner way to extract the parent topic description
            parent_name_series = topics_df[(topics_df['topic_key'] == parent_key) & (topics_df['parent_key'] == '')]['topic_name']
            parent_name = parent_name_series.values[0] # NOTE feels like there should be a cleaner way to extract the parent topic description

            topics_df.loc[(topics_df['parent_key'] == parent_key), 'parent_description'] = parent_desc
            topics_df.loc[(topics_df['parent_key'] == parent_key), 'parent_name'] = parent_name
        topics_df = topics_df.fillna('') # NOTE feels weird to do this a second time, but mop-up seems necessary in both places
    else:
        return {'Error': f'Unable to load topics for topic_grouping={topic_grouping}, no file found at file_path={file_path}'}
    
    es_topics = []
    for _, row in topics_df.iterrows():
        es_topic = EsTopic(topic_grouping=topic_grouping, topic_key=row['topic_key'], topic_name=row['topic_name'], description=row['description'])
        if 'parent_key' in row:
            es_topic.parent_key = row['parent_key']
        if 'parent_name' in row:
            es_topic.parent_name = row['parent_name']
        if 'parent_description' in row:
            es_topic.parent_description = row['parent_description']
        es_topics.append(es_topic)
    
    # write to es
    attempted_count = 0
    successful_topics = []
    failed_topics = []
    for es_topic in es_topics:
        attempted_count += 1
        try:
            esqb.save_es_topic(es_topic)
            successful_topics.append(es_topic.topic_key)
        except Exception as e:
            failed_topics.append(es_topic.topic_key)
            print(f'Failed to index es_topic.topic_key={es_topic.topic_key}: {e}')

    return {'attempted_count': attempted_count, 'successful_topics': successful_topics, 'failed_topics': failed_topics}


@esw_app.get("/populate_topic_embeddings/{topic_grouping}/{topic_key}/{model_vendor}/{model_version}")
def populate_topic_embeddings(topic_grouping: str, topic_key: str, model_vendor: str, model_version: str, user: user_dependency, prefix_parent_descs: bool = False):
    '''
    Generate vector embedding for topic using pre-trained Word2Vec and Transformer models
    '''
    exit_if_unauthorized(user, level='admin')
    
    embeddings_field = f'{model_vendor}_{model_version}_embeddings'
    doc_id = f'{topic_grouping}_{topic_key}'
    
    try:
        es_topic = EsTopic.get(id=doc_id)
        text_to_vectorize = es_topic.description
        if prefix_parent_descs and es_topic.parent_description:
            text_to_vectorize = f'{es_topic.parent_description} {text_to_vectorize}'
        embeddings = ef.generate_embeddings(text_to_vectorize, model_vendor, model_version)
        es_topic[embeddings_field] = embeddings
        esqb.save_es_topic(es_topic)
        return {"topic": es_topic._d_}
    except Exception as e:
        return {f"error": f"Failed to populate {model_vendor}:{model_version} embeddings for topic {topic_grouping}:{topic_key}, {e}"}
    

@esw_app.get("/populate_speaker_embeddings/{show_key}/{speaker}/{model_vendor}/{model_version}")
def populate_speaker_embeddings(show_key: ShowKey, speaker: str, model_vendor: str, model_version: str, user: user_dependency):
    '''
    Generate vector embedding for speaker using pre-trained Word2Vec and Transformer models
    '''
    exit_if_unauthorized(user, level='admin')
    
    max_tokens = TRF_MODELS[model_vendor]['versions'][model_version]['max_tokens']
    word_count_field = f'{model_vendor}_word_count'
    embeddings_field = f'{model_vendor}_{model_version}_embeddings'

    attempted_count = 0
    successful = []
    skipped = []
    failed = []
    failure_messages = []

    attempted_count += 1
    es_speaker_id = f'{show_key.value}_{speaker}'
    try:
        es_speaker = EsSpeaker.get(id=es_speaker_id)
    except Exception as e:
        return {"error": f"Failure to populate speaker embeddings, no match in `speakers` index for es_speaker_id={es_speaker_id}, {e}"}
    
    # vectorize and generate embeddings for es_speaker.lines, paring down text with shorten_lines_of_text if necessary/possible
    es_speaker_lines = es_speaker.lines
    if es_speaker[word_count_field] > max_tokens:
        es_speaker_lines = ef.shorten_lines_of_text(es_speaker_lines, max_tokens)
        if not es_speaker_lines:
            print(f"For speaker={speaker}, series {word_count_field}={es_speaker[word_count_field]} exceeds max_tokens={max_tokens} and attempts at shortening failed; skipping series-level embeddings")
            skipped.append(f'es_speaker_id={es_speaker_id}')
    if es_speaker_lines:
        print(f"Calling generate_embeddings on es_speaker_id={es_speaker_id} es_speaker[{word_count_field}]={es_speaker[word_count_field]}")
        try:
            text_to_vectorize = ' '.join(es_speaker_lines)
            embeddings = ef.generate_embeddings(text_to_vectorize, model_vendor, model_version)
            es_speaker[embeddings_field] = embeddings
            esqb.save_es_speaker(es_speaker)   
            successful.append(f'es_speaker_id={es_speaker_id}')     
        except Exception as e:
            return {f"error": f"Failed to populate {model_vendor}:{model_version} embeddings for speaker {show_key.value}:{es_speaker}, {e}"}
    
    # iterate through speaker_seasons indexed in es_speaker.seasons_to_episode_keys
    for season, episode_keys in es_speaker.seasons_to_episode_keys._d_.items():
        attempted_count += 1
        es_speaker_season_id = f'{show_key.value}_{speaker}_{season}'
        try:
            es_speaker_season = EsSpeakerSeason.get(id=es_speaker_season_id)
        except Exception as e:
            err = f"Failure to fetch EsSpeakerSeason with id={es_speaker_season_id}: {e}"
            print(err)
            failure_messages.append(err)
            failed.append(f'es_speaker_season_id={es_speaker_season_id}')
            continue

        # vectorize and generate embeddings for es_speaker_season.lines, paring down text with shorten_lines_of_text if necessary/possible
        es_speaker_season_lines = es_speaker_season.lines
        if es_speaker_season[word_count_field] > max_tokens:
            es_speaker_season_lines = ef.shorten_lines_of_text(es_speaker_season_lines, max_tokens)
            if not es_speaker_season_lines:
                print(f"For es_speaker_season_id={es_speaker_season_id}, es_speaker_season[{word_count_field}]={es_speaker_season[word_count_field]} exceeds max_tokens={max_tokens} and attempts at shortening failed; skipping season-level embeddings for season={season}")
                skipped.append(f'es_speaker_season_id={es_speaker_season_id}')
        if es_speaker_season_lines:
            print(f"Calling generate_embeddings on es_speaker_season_id={es_speaker_season_id} es_speaker_season[{word_count_field}]={es_speaker_season[word_count_field]}")
            try:
                text_to_vectorize = ' '.join(es_speaker_season_lines)
                embeddings = ef.generate_embeddings(text_to_vectorize, model_vendor, model_version)
                es_speaker_season[embeddings_field] = embeddings
                esqb.save_es_speaker_season(es_speaker_season)
                successful.append(f'es_speaker_season_id={es_speaker_season_id}')      
            except Exception as e:
                err = f"Failed to populate {model_vendor}:{model_version} embeddings for es_speaker_season_id={es_speaker_season_id}: {e}"
                print(err)
                failure_messages.append(err)
                failed.append(f'es_speaker_season_id={es_speaker_season_id}')

        # iterate through speaker_episodes indexed in es_speaker.seasons_to_episode_keys
        for episode_key in episode_keys:
            attempted_count += 1
            es_speaker_episode_id = f'{show_key.value}_{speaker}_{episode_key}'
            try:
                es_speaker_episode = EsSpeakerEpisode.get(id=es_speaker_episode_id)
            except Exception as e:
                err = f"Failure to fetch EsSpeakerEpisode with id={es_speaker_episode_id}: {e}"
                print(err)
                failure_messages.append(err)
                failed.append(f'es_speaker_episode_id={es_speaker_episode_id}')
                continue

            # vectorize and generate embeddings for es_speaker_episode.lines, paring down text with shorten_lines_of_text if necessary/possible
            es_speaker_episode_lines = es_speaker_episode.lines
            if es_speaker_episode[word_count_field] > max_tokens:
                es_speaker_episode_lines = ef.shorten_lines_of_text(es_speaker_episode_lines, max_tokens)
                if not es_speaker_episode_lines:
                    print(f"For es_speaker_episode_id={es_speaker_episode_id}, es_speaker_episode[{word_count_field}]={es_speaker_episode[word_count_field]} exceeds max_tokens={max_tokens} and attempts at shortening failed; skipping embeddings for episode_key={episode_key}")
                    skipped.append(f'es_speaker_episode_id={es_speaker_episode_id}')
            if es_speaker_episode_lines:
                print(f"Calling generate_embeddings on es_speaker_episode_id={es_speaker_episode_id} es_speaker_episode[{word_count_field}]={es_speaker_episode[word_count_field]}")
                try:
                    text_to_vectorize = ' '.join(es_speaker_episode_lines)
                    embeddings = ef.generate_embeddings(text_to_vectorize, model_vendor, model_version)
                    es_speaker_episode[embeddings_field] = embeddings
                    esqb.save_es_speaker_episode(es_speaker_episode)
                    successful.append(f'es_speaker_episode_id={es_speaker_episode_id}')      
                except Exception as e:
                    err = f"Failed to populate {model_vendor}:{model_version} embeddings for es_speaker_episode_id={es_speaker_episode_id}: {e}"
                    print(err)
                    failure_messages.append(err)
                    failed.append(f'es_speaker_episode_id={es_speaker_episode_id}')

    return {'attempted_count': attempted_count, 'successful': successful, 'skipped': skipped, 'failed': failed, 'failure_messages': failure_messages}


@esw_app.get("/populate_episode_topics/{show_key}/{episode_key}/{topic_grouping}/{model_vendor}/{model_version}")
def populate_episode_topics(show_key: ShowKey, episode_key: str, topic_grouping: str, model_vendor: str, model_version: str, user: user_dependency):
    '''
    Generate and store topic mappings for episode, via knn cosine similarity to vector embeddings within topic_grouping 
    '''
    exit_if_unauthorized(user, level='admin')
    
    es_episode = EsEpisodeTranscript.get(id=f'{show_key.value}_{episode_key}')
    try:
        response = esr.episode_topic_vector_search(show_key, episode_key, topic_grouping, model_vendor=model_vendor, model_version=model_version)
        if 'topics' not in response:
            return {"error": f"Failed to populate_episode_topics, episode_topic_vector_search returned no topics for {show_key.value}:{episode_key} topic_grouping={topic_grouping} model={model_vendor}:{model_version}"}
    except Exception as e:
        return {"error": f"Failed to populate_episode_topics, episode_topic_vector_search failed for {show_key.value}:{episode_key} topic_grouping={topic_grouping} model={model_vendor}:{model_version}: {e}"}
    
    # write to episode_topics
    episode_topics = esqb.populate_episode_topics(show_key.value, es_episode, response['topics'], model_vendor, model_version)

    # write simplified subset of episode_topics to es_episode.topics_X
    simple_episode_topics = fflat.flatten_es_topics(episode_topics)
    # TODO this is out of date, topics_universal should either be topics_universal_{model_name} or it should be a dict keying off of model_name instead of a list of topics
    # For now I'm only indexing topics generated with openai:3small embeddings 
    if model_vendor == 'openai' and model_version == '3small':
        if topic_grouping == 'universalGenres':
            es_episode.topics_universal = simple_episode_topics
        # elif topic_grouping == 'focusedGpt35_TNG':
        #     es_episode.topics_focused = simple_episode_topics
    esqb.save_es_episode(es_episode)

    return {"episode_topics": episode_topics}


@esw_app.post("/populate_episode_tfidf_topics")
def populate_episode_tfidf_topics(show_key: ShowKey, episode_key: str, topic_grouping: str, simple_episode_topics: list, user: user_dependency):
    '''
    Store tfidf-style topic mappings for episode 
    '''
    exit_if_unauthorized(user, level='admin')
    
    episode_id = f'{show_key.value}_{episode_key}'
    if not simple_episode_topics:
        print(f'Failed to populate_episode_tfidf_topics for episode {episode_id}, simple_episode_topics list was empty')
        return {}
    
    es_episode = EsEpisodeTranscript.get(id=episode_id)
    if topic_grouping == 'universalGenres':
        es_episode.topics_universal_tfidf = simple_episode_topics
    # elif topic_grouping == 'focusedGpt35_TNG':
    #     es_episode.topics_focused_tfidf = simple_episode_topics
    es_episode.save()

    return {"es_episode": episode_id}


@esw_app.get("/populate_speaker_topics/{show_key}/{speaker}/{topic_grouping}/{model_vendor}/{model_version}")
def populate_speaker_topics(show_key: ShowKey, speaker: str, topic_grouping: str, model_vendor: str, model_version: str, user: user_dependency):
    '''
    Using previously generated vector embeddings for speakers and topics, use knn vector cosine similarity to map speakers to topics, then populate speaker indexes with topics
    Populate speaker topics at series-, season-, and episode-level, using vector embeddings at each level where possible (when text corpus is small enough for embeddings generation)
    When a series- or season-level text corpus is too large for its own embedding, use topics mapped to sub-elements (episodes in season, seasons in series) to aggregate topic mappings
    '''
    exit_if_unauthorized(user, level='admin')
    
    es_speaker = EsSpeaker.get(id=f'{show_key.value}_{speaker}')

    topic_fields = 'topic_grouping,topic_key,parent_key,topic_name,parent_name'
    reference_topics_response = esr.fetch_topic_grouping(topic_grouping, return_fields=topic_fields)
    if 'topics' not in reference_topics_response or len(reference_topics_response['topics']) == 0:
        return {"error": f"Failure to populate_speaker_topics: no topics returned from /fetch_topic_grouping for topic_grouping={topic_grouping}"}
    reference_topics = {t['topic_key']:t for t in reference_topics_response['topics']}

    speaker_topics_response = esr.speaker_topic_vector_search(show_key, speaker, topic_grouping, model_vendor=model_vendor, model_version=model_version)
    # TODO since `speaker_topic_vector_search` is shared functionality, do I need to verify that a full response was generated before writing?

    if not all(k in speaker_topics_response for k in ('series_topics', 'season_topics', 'episode_topics')):
        err = f"Failure to populate_speaker_topics: response from `speaker_topic_vector_search` must include 'series_topics', 'season_topics', and 'episode_topics'"
        print(f"{err}, incomplete topics_resopnse={speaker_topics_response}")
        return {"error": err, "incomplete topics_resopnse": speaker_topics_response}
    
    speaker_series_topics = speaker_topics_response['series_topics']
    speaker_topics_by_season = speaker_topics_response['season_topics']
    speaker_topics_by_episode = speaker_topics_response['episode_topics']

    series_topics_found = False
    series_topic_agg = TopicAgg(reference_topics)
    if len(speaker_series_topics) > 0:
        series_topics_found = True
        
    for season, episode_keys in es_speaker.seasons_to_episode_keys._d_.items():
        season = int(season)
        season_topics_found = False
        season_topic_agg = TopicAgg(reference_topics)
        es_speaker_season = EsSpeakerSeason.get(id=f'{show_key.value}_{speaker}_{season}')
        if season in speaker_topics_by_season:
            speaker_season_topics = speaker_topics_by_season[season]
            season_topics_found = True
        for e_key in episode_keys:
            if e_key in speaker_topics_by_episode:
                es_speaker_episode = EsSpeakerEpisode.get(id=f'{show_key.value}_{speaker}_{e_key}')
                # write to speaker_episode_topics
                es_speaker_episode_topics = esqb.populate_speaker_episode_topics(show_key.value, speaker, es_speaker_episode, speaker_topics_by_episode[e_key],
                                                                                 model_vendor, model_version)
                
                # write simplified subset of episode_topics to es_speaker_episode.topics_X
                # TODO this is out of date, topics_mbti and topics_dnda should either be topics_{type}_{model_name} or it should be a dict keying off of model_name instead of a list of topics
                # For now I'm only indexing topics generated with openai:3small embeddings 
                if model_vendor == 'openai' and model_version == '3small':
                    simple_episode_topics = fflat.flatten_es_topics(es_speaker_episode_topics)
                    if topic_grouping == 'mbti':
                        es_speaker_episode.topics_mbti = simple_episode_topics
                    elif topic_grouping == 'dndAlignments':
                        es_speaker_episode.topics_dnda = simple_episode_topics
                    esqb.save_es_speaker_episode(es_speaker_episode)

                # incorporate episode topics into season-level agg
                season_topic_agg.add_topics(speaker_topics_by_episode[e_key], es_speaker_episode.word_count)
            else:
                print(f"Warning: episode_key={e_key} found in `es_speaker.seasons_to_episode_keys` but not in `speaker_topics_response['episode_topics']`, skipping but this is weird")
        
        # write to speaker_season_topics
        if not season_topics_found:
            # if no season-level topics, attempt to calculate them via aggs extracted from episodes
            print(f'no season_topics_found for season={season}, so using topics from season_topic_agg={season_topic_agg}')
            speaker_season_topics = season_topic_agg.get_topics()
        es_speaker_season_topics = esqb.populate_speaker_season_topics(show_key.value, speaker, es_speaker_season, speaker_season_topics, model_vendor, model_version)

        # write simplified subset of season_topics to es_speaker_season.topics_X
        # TODO this is out of date, topics_mbti and topics_dnda should either be topics_{type}_{model_name} or it should be a dict keying off of model_name instead of a list of topics
        # For now I'm only indexing topics generated with openai:3small embeddings 
        if model_vendor == 'openai' and model_version == '3small':
            simple_season_topics = fflat.flatten_es_topics(es_speaker_season_topics)
            if topic_grouping == 'mbti':
                es_speaker_season.topics_mbti = simple_season_topics
            elif topic_grouping == 'dndAlignments':
                es_speaker_season.topics_dnda = simple_season_topics
            esqb.save_es_speaker_season(es_speaker_season)

        # incorporate season topics into series-level agg
        series_topic_agg.add_topics(speaker_season_topics, es_speaker_season.word_count)

    # write to speaker_topics
    if not series_topics_found:
        # if no series-level topics, attempt to calculate them via aggs extracted from seasons  
        print(f'no series_topics_found, so using topics from series_topic_agg={series_topic_agg}')
        speaker_series_topics = series_topic_agg.get_topics()
    print(f'speaker_series_topics={speaker_series_topics}')
    es_speaker_topics = esqb.populate_speaker_topics(show_key.value, speaker, es_speaker, speaker_series_topics, model_vendor, model_version)
    
    # write simplified subset of speaker_topics to es_speaker.topics_X
    # TODO this is out of date, topics_mbti and topics_dnda should either be topics_{type}_{model_name} or it should be a dict keying off of model_name instead of a list of topics
    # For now I'm only indexing topics generated with openai:3small embeddings 
    if model_vendor == 'openai' and model_version == '3small':
        simple_series_topics = fflat.flatten_es_topics(es_speaker_topics)
        if topic_grouping == 'mbti':
            es_speaker.topics_mbti = simple_series_topics
        elif topic_grouping == 'dndAlignments':
            es_speaker.topics_dnda = simple_series_topics
        esqb.save_es_speaker(es_speaker)

    # TODO ugh these's caching or latency with these lookups, responses are stale
    speaker_topics_response = esr.fetch_speaker_topics(speaker, show_key, topic_grouping)
    speaker_season_topics_response = esr.fetch_speaker_season_topics(show_key, topic_grouping, speaker=speaker, limit=1000)
    speaker_episode_topics_response = esr.fetch_speaker_episode_topics(show_key, topic_grouping, speaker=speaker, limit=10000)

    return {"speaker_topics": speaker_topics_response['speaker_topics'], 
            "speaker_season_topics": speaker_season_topics_response['speaker_season_topics'], 
            "speaker_episode_topics": speaker_episode_topics_response['speaker_episode_topics']}


@esw_app.get("/populate_episode_narratives/{show_key}/{episode_key}")
def populate_episode_narratives(show_key: ShowKey, episode_key: str, user: user_dependency):
    '''
    Generate and populate narrative sequences for a given episode
    '''
    exit_if_unauthorized(user, level='admin')
    
    narrative_sequences = ne.extract_narrative_sequences(show_key, episode_key)

    for ns in narrative_sequences:
        # irritating datatype conversion for scene indexes, elasticsearch won't accept int keys in its Object type
        source_scene_wcs = {str(k):v for k,v in ns['source_scene_wcs'].items()}
        es_narrative_sequence = EsEpisodeNarrativeSequence(show_key=show_key.value, episode_key=episode_key, speaker_group='_'.join(ns['speaker_group']),
                                                           narrative_lines=ns['narrative_lines'], word_count=int(ns['wc']),
                                                           source_scene_word_counts=source_scene_wcs, speaker_line_counts=ns['speaker_line_counts'])
        es_narrative_sequence.save()

    return {"narrative_sequences": narrative_sequences}




# @esw_app.get("/populate_episode_polarity_sentiment/{show_key}/{episode_key}")
# def populate_episode_polarity_sentiment(show_key: ShowKey, episode_key: str, scene_level: bool = False, scene_event_level: bool = False):
#     '''
#     Generate and populate nltk polarity sentiment for episode
#     '''
#     es_episode = EsEpisodeTranscript.get(id=f'{show_key.value}_{episode_key}')
#     scene_sentiments = []
#     if scene_level:
#         flattened_scenes_response = esr.fetch_flattened_scenes(show_key, episode_key)
#         flattened_scenes = flattened_scenes_response['flattened_scenes']
#         # print(f'len(flattened_scenes)={len(flattened_scenes)}')
#         for i in range(len(flattened_scenes)):
#             scene_sentiment = sa.generate_polarity_sentiment(flattened_scenes[i])
#             es_episode.scenes[i].nltk_sent_pos = scene_sentiment['pos']
#             es_episode.scenes[i].nltk_sent_neg = scene_sentiment['neg']
#             es_episode.scenes[i].nltk_sent_neu = scene_sentiment['neu']
#             scene_sentiments.append((flattened_scenes[i], dict(pos=scene_sentiment['pos'], neg=scene_sentiment['neg'], neu=scene_sentiment['neu'])))
#     if scene_event_level:
#         for es_scene in es_episode.scenes:
#             # agg_wc = 0
#             # agg_pos = 0
#             # agg_neg = 0
#             # agg_neu = 0
#             # speaker_sentiments = {}
#             # speaker_word_counts = {}
#             for es_scene_event in es_scene.scene_events:
#                 if es_scene_event.spoken_by and es_scene_event.dialog:
#                     line_sentiment = sa.generate_polarity_sentiment(es_scene_event.dialog)
#                     # set line-level sentiment
#                     es_scene_event.nltk_sent_pos = line_sentiment['pos']
#                     es_scene_event.nltk_sent_neg = line_sentiment['neg']
#                     es_scene_event.nltk_sent_neu = line_sentiment['neu']
#                     # # for aggregation and eventual averaging, calculate weighted sentiment based on word count
#                     # line_wc = es_scene_event.dialog.word_count
#                     # line_speaker = es_scene_event.spoken_by
#                     # line_weighted_pos = line_sentiment['pos'] * line_wc
#                     # line_weighted_neg = line_sentiment['neg'] * line_wc
#                     # line_weighted_neu = line_sentiment['neu'] * line_wc
#                     # # aggregate scene-level sentiment
#                     # agg_wc += line_wc
#                     # agg_pos += line_weighted_pos
#                     # agg_neg += line_weighted_neg
#                     # agg_neu += line_weighted_neu
#                     # # aggregate scene-level speaker sentiment
#                     # if line_speaker not in speaker_sentiments:
#                     #     speaker_sentiments[line_speaker] = dict(nltk_sent_pos=0, nltk_sent_neg=0, nltk_sent_neu=0)
#                     # speaker_sentiments[line_speaker]['nltk_sent_pos'] += line_weighted_pos
#                     # speaker_sentiments[line_speaker]['nltk_sent_neg'] += line_weighted_neg
#                     # speaker_sentiments[line_speaker]['nltk_sent_neu'] += line_weighted_neu
#                     # speaker_word_counts += line_wc

#     episode_sentiment = sa.generate_polarity_sentiment(es_episode.flattened_text)
#     esqb.save_episode_sentiment(es_episode, episode_sentiment)

#     return {f"title": es_episode.title,
#              "season": es_episode.season, 
#              "episode": es_episode.sequence_in_season, 
#              "episode_sentiment": episode_sentiment,
#              "scene_sentiments": scene_sentiments}


# @esw_app.get("/populate_episode_emotional_sentiment/{show_key}/{episode_key}")
# def populate_episode_emotional_sentiment(show_key: ShowKey, episode_key: str, scene_level: bool = False, line_level: bool = False, write_to_es: bool = False):
#     '''
#     Generate and populate openai emotional sentiment for episode. Currently populating to 3 places: 
#     1. api response
#     2. dataframe
#     3. es index (optional)
#     '''
#     es_episode = EsEpisodeTranscript.get(id=f'{show_key.value}_{episode_key}')

#     openai_total_reqs = 0
#     openai_success_reqs = 0
#     openai_failure_reqs = []
#     start_ts = time.time()
#     print(f'begin generate_emotional_sentiment against full episode at start_ts={start_ts}')

#     # episode-level emotional sentiment 
#     openai_total_reqs += 1
#     episode_emo_df, episode_emo_dict = sa.generate_emotional_sentiment(es_episode.flattened_text)
#     if episode_emo_df is None:
#         return {"error": f"failure to execute generate_emotional_sentiment on es_episode.flattened_text for show_key={show_key.value} episode_key={episode_key}"}
#     openai_success_reqs += 1
#     # add contextual properties to emo_df
#     episode_emo_df['key'] = 'E'
#     episode_emo_df['scene'] = 'ALL'
#     episode_emo_df['line'] = 'ALL'
#     episode_emo_df['speaker'] = 'ALL'
#     # update es object
#     if write_to_es:
#         for emo in OPENAI_EMOTIONS:
#             set_dict_value_as_es_value(es_episode, episode_emo_dict, emo, 'openai_sent_')

#     # scene- and line-level emotional sentiment 
#     scene_emo_dicts = []
#     if scene_level or line_level:

#         # scene-level processing will use fetch_flattened_scenes, trusting (gulp) that scene index positions align with their es_episode.scenes counterparts
#         if scene_level:
#             flattened_scenes_response = esr.fetch_flattened_scenes(show_key, episode_key)
#             flattened_scenes = flattened_scenes_response['flattened_scenes']

#         # both scene- and line-level processing iterate over es_episode.scenes, carefully tracking scene index position
#         for scene_i in range(len(es_episode.scenes)):
#             es_scene = es_episode.scenes[scene_i]
#             scene_emo_dict = dict(scene_i=scene_i, scene_level=None, line_level=[])
#             scene_emo_dicts.append(scene_emo_dict)

#             # scene-level: analyze flattened_scene
#             if scene_level:
#                 print(f'executing generate_emotional_sentiment on flattened_scene at scene_i={scene_i}')
#                 if not flattened_scenes[scene_i]:
#                     print(f'flattened_scene at scene_i={scene_i} contains no dialog text, skipping')
#                     continue
#                 openai_total_reqs += 1
#                 scene_emo_df, scene_emo_dict['scene_level'] = sa.generate_emotional_sentiment(flattened_scenes[scene_i])
#                 if scene_emo_df is None:
#                     failure_message = f'failure to execute generate_emotional_sentiment on flattened_scene at scene_i={scene_i} with text=`{flattened_scenes[scene_i]}`'
#                     openai_failure_reqs.append(failure_message)
#                     print(failure_message)
#                     continue
#                 openai_success_reqs += 1
#                 scene_emo_df['key'] = f'S{scene_i}'
#                 scene_emo_df['scene'] = scene_i
#                 scene_emo_df['line'] = 'ALL'
#                 scene_emo_df['speaker'] = 'ALL'
#                 episode_emo_df = pd.concat([episode_emo_df, scene_emo_df], axis=0)
#                 # update es object
#                 if write_to_es:
#                     for emo in OPENAI_EMOTIONS:
#                         set_dict_value_as_es_value(es_scene, episode_emo_dict, emo, 'openai_sent_')

#             # line-level: analyze dialog for each line in scene
#             if line_level:
#                 line_i = 0
#                 for es_scene_event in es_scene.scene_events:
#                     if es_scene_event.spoken_by and es_scene_event.dialog:
#                         print(f'executing generate_emotional_sentiment on flattened_scene at scene_i={scene_i} line_i={line_i}')
#                         openai_total_reqs += 1
#                         line_emo_df, line_emo_dict = sa.generate_emotional_sentiment(es_scene_event.dialog)
#                         if line_emo_df is None:
#                             failure_message = f'failure to execute generate_emotional_sentiment on flattened_scene at scene_i={scene_i} line_i={line_i} es_scene_event.dialog=`{es_scene_event.dialog}`'
#                             openai_failure_reqs.append(failure_message)
#                             print(failure_message)
#                             continue
#                         openai_success_reqs += 1
#                         scene_emo_dict['line_level'].append(line_emo_dict)
#                         line_emo_df['key'] = f'S{scene_i}L{line_i}'
#                         line_emo_df['scene'] = scene_i
#                         line_emo_df['line'] = line_i
#                         line_emo_df['speaker'] = es_scene_event.spoken_by
#                         line_i += 1
#                         episode_emo_df = pd.concat([episode_emo_df, line_emo_df], axis=0)
#                         # update es object
#                         if write_to_es:
#                             for emo in OPENAI_EMOTIONS:
#                                 set_dict_value_as_es_value(es_scene_event, episode_emo_dict, emo, 'openai_sent_')

#     end_ts = time.time()
#     duration = end_ts - start_ts
#     duration = round(duration, 2)
#     print(f'finish generate_emotional_sentiment against full episode at end_ts={end_ts}')

#     # write dataframe to csv
#     file_path = f'{PATH_TO_SENTIMENT_DATA}/{show_key.value}/{show_key.value}_{episode_key}.csv'
#     episode_emo_df.to_csv(file_path, sep=',', header=True)

#     # write to es
#     if write_to_es:
#         esqb.save_es_episode(es_episode)

#     return {"duration": duration, 
#             "openai_total_reqs": openai_total_reqs,
#             "openai_success_reqs": openai_success_reqs,
#             "openai_failure_reqs": openai_failure_reqs,
#             "episode_emo_dict": episode_emo_dict, 
#             "scene_emo_dicts": scene_emo_dicts}


# @esw_app.get("/test_episode_emotional_sentiment")
# def test_episode_emotional_sentiment():
    
#     flattened_scenes = ["PICARD: Captain's log, stardate 43930.7. The Enterprise has been in attendance at the biennial Trade Agreements Conference on Betazed. For the first time, the Ferengi are present, and I have reluctantly consented to their boarding the Enterprise for the closing reception.\n\nRIKER: Check and mate.\n\nWESLEY: Perfect. The queen's gambit finished off with the Aldabren Exchange.\n\nNIBOR: That is unfair. I couldn't concentrate with all that noise.\n\nWESLEY: Noise? It's Algolian ceremonial rhythms.\n\nPICARD: A toast to the success of the trade conference, Reittan. I must admit, I had some doubts when you invited the Ferengi.\n\nGRAX: They made a profit and behaved themselves. What more could one ask? Still, they trouble me. We Betazeds are uncomfortable with species like the Ferengi whose minds we can't read.\n\nDATA: Perhaps your telepathic abilities are ineffective owing to the anomalous construction of the Ferengi brain, which is composed of four different\n\nPICARD: Thank you, Mister Data. It was thoughtful of you to invite Lwaxana Troi to be part of the Betazed delegation.\n\nGRAX: Yes, Lwaxana and I go way back. Her first husband and I were old friends, and I've known Deanna since she was a child.\n\nPICARD: I'm sure Counsellor Troi appreciates the opportunity to spend time with her mother.\n\nLWAXANA: Little One, you could at least pretend you're happy to see me.\n\nTROI: Mother, we're among non-telepaths. It's impolite not to speak aloud.\n\nLWAXANA: You mean talk with my mouth full? Deanna, please!\n\nLWAXANA: All right, you want me to say something aloud? Have you considered if you had stayed on Betazed, you might have been a happier person.\n\nTROI: Let's not guess what might have been. I love my work aboard the Enterprise.\n\nLWAXANA: Yes, of course you do, but its all business and no play. You've got to enjoy life, relax, like I do. Find yourself the right man, think of your future. Think of my future.\n\nRIKER: Lwaxana, Deanna. Anything I can do for you?\n\nTROI: Could I?\n\nFAREK: She's as repulsive as the rest of them.\n\nTOG: Repulsive? I find her exotic. And what an advantage her telepathy would be in our negotiations.\n\nFAREK: To read our competitors' minds? Yes, that would be valuable. But she'd never agree to use her powers to help us.\n\nTOG: I'm not so sure.\n\nTOG: Lwaxana Troi of Betazed, I believe. I am DaiMon Tog of the Ferengi vessel Krayton. May I join you?\n\nLWAXANA: I was just going to see Captain Picard. Excuse me.\n\nLWAXANA: Oh, Jean-Luc! Jean-Luc! Come have a drink with me. Tell me what you've been up to.\n\nPICARD: Perhaps later, Lwaxana. Mister Data and I were just about to show Reittan Grax the er, the er, the new door mechanisms on the aft turbolifts. If you'll excuse us?\n\nTOG: Lwaxana Troi. I desire you.\n\nLWAXANA: What?\n\nTOG: You see, your Betazoid skills would be very useful to me, and I find you very attractive. I am willing to pay handsomely for you.\n\nLWAXANA: I don't believe this.\n\nTOG: You must be aware that every female has her price.\n\nLWAXANA: Let's get one thing straight, little man. I am not for sale. And if, by some chance I were to become available, I would rather eat Orion wing-slugs than deal with a toad-faced troll like you! So go away and find someone else to become your property.\n\nTOG: As you wish.\n\nFAREK: Now that you've totally humiliated us, may we return to our vessel?\n\nTOG: She is exhilarating, isn't she? Now I want her more than ever. Lwaxana Troi, you will be mine.",
#         "TROI: Are we at war with the Ferengi yet?\n\nWORF: DaiMon Tog has returned to his vessel and the Ferengi have left orbit.\n\nTROI: My mother will be relieved.\n\nWORF: I hear she handled the situation quite skillfully. An admirable woman.\n\nTROI: I'll be sure to tell her you said so.\n\nLWAXANA: Come in, Little One.",
#         "TROI: Mother? Mother, please.\n\nLWAXANA: After that awful little Ferengi insulted me, I needed to centre myself. Can you imagine that dreadful little creature talking to me like that? Doesn't he realise that I am a daughter of the Fifth House of Betazed. Holder of the Sacred Chalice of Rixx?\n\nTROI: The Sacred Chalice of Rixx is an old clay pot with mold growing inside it.\n\nLWAXANA: Perhaps one day when you're older and wiser, you'll understand. Come on, sit down. Talk to me. We spend so little time together.\n\nTROI: That's true. I'm sorry, and I didn't mean to get so upset with you at the reception.\n\nLWAXANA: Deanna, try to understand. You're all I have. My only concern is for your happiness.\n\nTROI: I am happy. Why can't you believe that?\n\nLWAXANA: I wish I could, but how much happiness is there in always being there for someone else, and never being there for yourself?\n\nTROI: I get a great deal of satisfaction out of my work.\n\nLWAXANA: I'm sure you do. I'm sure it's very rewarding in its way. What about a family?\n\nTROI: This is my family. My friends here on the Enterprise.\n\nLWAXANA: All right. In case I have to spell it out for you, I'm talking about finding a husband, having a child. That's what made me happy. At least until now.\n\nTROI: Mother, look. Perhaps some day I will marry. But you've got to let me make my own choices, live my own life, and not the life you would choose for me.\n\nLWAXANA: You had your chance with Commander Riker. Look how you ruined that.\n\nTROI: I did not ruin anything. We've became very good friends.\n\nLWAXANA: Well, all the better. You certainly wouldn't want to marry an enemy. I see we can't talk about this. Very well, have it your way, Little One.\n\nTROI: Little One? You called me that when I was five. Now stop demeaning me and address me as an adult!",
#         "LWAXANA: I'll be home on Betazed if you need me, Little. Deanna.",
#         "WESLEY: Adding pre-processors to the neutrino counters boosted efficiency by eleven percent. Of course, Commander La Forge and Commander Data did most of the work.\n\nLAFORGE: Not so fast, Wesley. Pre-processing the data with an optical chip was your idea.\n\nDATA: That is correct, sir. Although Commander La Forge and I designed the chip, Mister Crusher derived the equation governing its operation.\n\nRIKER: The point is, you've completed the upgrade well ahead of schedule. Very impressive.\n\nPICARD: Indeed. Fine work. We shall miss you, Mister Crusher. As you've guessed, final entrance examination scores from Starfleet Academy have arrived. Congratulations. As soon as you have completed the oral exam, you'll be formally admitted.\n\nRIKER: We'll have you back at Betazed in plenty of time to meet up with the Academy transport ship.\n\nWESLEY: Thank you, sir.\n\nPICARD: That'll be all. Number One, a moment.\n\nRIKER: Yes, Captain?\n\nPICARD: You'll agree that this is a fairly routine mapping mission?\n\nRIKER: Yes, sir.\n\nPICARD: Counsellor Troi had the good sense to ask for shore leave. I can see I'm going to have to suggest it to you. Have a good time, Number One.",
#         "RIKER: I think it's around here somewhere.\n\nTROI: Maybe it died. It has been a few years.\n\nRIKER: Muktok live for hundreds of years. Here it is.\n\nTROI: It's lovely. I remember that sound, and all the good times we had.\n\nRIKER: I remember a certain junior officer meeting a very serious psychologist. The best part about being assigned to Betazed.\n\nLWAXANA: Oh, this is the perfect spot. Put the food down over there, Mister Homn. Isn't it a beautiful day for a picnic? They'll join us in a minute. No, no, no, no. Here, put the food over there. No, you can go back\n\nTROI: Mother, how did you know about this place?\n\nLWAXANA: Your father used to bring me here. Sit down.\n\nRIKER: And you even brought provisions. Very thoughtful.\n\nLWAXANA: Here, Will, Deanna. Try an oskoid. They're delicious. That sap running through the veins helps keep it warm.\n\nRIKER: Very tasty. So tell us, Lwaxana. The last time we met, you were looking for a husband. Did you have any luck?\n\nLWAXANA: Alas, no, but what happens to me isn't important. I'm much more concerned about other people getting on with their lives.\n\nTROI: Mother.\n\nLWAXANA: Mister Homn, I noticed some uttaberries back along the path. Pick some.\n\nLWAXANA: Well, Mister Homn and I could go back home if you two would like to be alone. It's such a romantic setting.\n\nTROI: Mother, stop it.\n\nLWAXANA: Darling, you have been so excitable lately. Have you ever thought of a leave of absence? I could talk to Jean-Luc.\n\nRIKER: Try the oskoid. Very different.\n\nRIKER: What the?\n\nTOG: For one whose beauty surpasses even these pericules.\n\nRIKER: DaiMon Tog, I thought the Krayton left orbit hours ago.\n\nTOG: It did. But when I tried to get the image of Lwaxana Troi out of my mind, I could not succeed.\n\nLWAXANA: This is ludicrous. You mean you came all the way back to Betazed for me?\n\nTOG: Why continue to search for perfection once you have found it?\n\nTROI: I don't believe this.\n\nLWAXANA: Look, Demon Tog, or whatever you call yourself, I am the Daughter of the Fifth House, Holder of the Sacred Chalice of Rixx, heir to the Holy Rings of Betazed. And unless you want to create an interstellar incident, you had better beam back to your ship.\n\nTOG: Returning to my ship is exactly what I had in mind. Krayton, transport four immediately.\n\nRIKER: No, Tog!",]
    
#     for scene in flattened_scenes:
#         _, _ = sa.generate_sentiment(scene, 'openai_emo', multi_speaker=True)
    
#     return {'success': 'success'}
