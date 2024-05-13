# import gensim
from gensim.models import Word2Vec, KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import common_texts
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openai
from openai import OpenAI
import os
import pandas as pd
from sklearn.cluster import KMeans
import tiktoken
import warnings

from app.config import settings
from app.es.es_model import EsEpisodeTranscript
import app.es.es_read_router as esr
from app.nlp.nlp_metadata import WORD2VEC_VENDOR_VERSIONS as W2V_MODELS, TRANSFORMER_VENDOR_VERSIONS as TRF_MODELS
import app.nlp.query_preprocessor as qpp
from app.show_metadata import ShowKey


warnings.filterwarnings(action = 'ignore')


cached_models = {}


# async def load_model(vendor: str, version: str) -> Word2Vec:
#     model_path = f'./{vendor}/{version}/model.txt'
#     model = Word2Vec.load(model_path)
#     print(f'loaded model={model} model_path={model_path} type(model)={type(model)}')
#     return model


'''
https://stackoverflow.com/questions/45310409/using-a-word2vec-model-pre-trained-on-wikipedia
http://vectors.nlpl.eu/explore/embeddings/en/models/
https://fasttext.cc/docs/en/english-vectors.html
https://code.google.com/archive/p/word2vec/
https://nlp.stanford.edu/projects/glove/
'''
def load_keyed_vectors(vendor: str, version: str) -> KeyedVectors:
    vendor_meta = W2V_MODELS[vendor]
    no_header = vendor_meta['no_header']
    file_suffix = vendor_meta['file_suffix']
    model_path = f'./w2v_models/{vendor}/{version}{file_suffix}'
    # TODO refactor into model metadata
    # if vendor == 'glove':
    #     no_header = True
    # else:
    #     no_header = False
    # if vendor == 'fasttext':
    #     model_path = f'./w2v_models/{vendor}/{version}.vec'
    # else:
    #     model_path = f'./w2v_models/{vendor}/{version}_model.txt'
    
    if model_path in cached_models:
        print(f'found model_path={model_path} in previously loaded models')
        word_vectors = cached_models[model_path]
    else:
        print(f'did not find model_path={model_path} in previously loaded models, loading now...')
        # if vendor == 'glove':
        #     glove2word2vec(glove_file, tmp_file) 
        word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False, no_header=no_header)
        cached_models[model_path] = word_vectors
        print(f'model_path={model_path} len(word_vectors)={len(word_vectors)} type(word_vectors)={type(word_vectors)}')
    # print(word_vectors.most_similar("vacation_NOUN"))
    # print(word_vectors.most_similar(positive=['woman_NOUN', 'king_NOUN'], negative=['man_NOUN']))

    return word_vectors


# '''
# from fasttext site
# '''
# def load_vectors(fname):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(float, tokens[1:])
#     return data


def calculate_embeddings(token_arr: list, model_vendor: str, model_version: str) -> tuple[list, list, list]:
    print('------------------------------------------------------------------------------------')
    print(f'begin calculate_embeddings using model {model_vendor}:{model_version} token_arr={token_arr}')

    vendor_meta = W2V_MODELS[model_vendor]
    embedding_sum = [0.0] * vendor_meta['versions'][model_version]['dims']

    keyed_vectors = load_keyed_vectors(model_vendor, model_version)

    tokens_processed = []
    tokens_failed = []
    for token in token_arr:
        if token not in keyed_vectors.key_to_index:
            # print(f'did not find token={token}, skipping')
            tokens_failed.append(token)
        else:
            embedding_sum += keyed_vectors.get_vector(token)
            tokens_processed.append(token)
    
    if len(tokens_processed) == 0:
        raise Exception('No tokens processed, cannot calculate embedding avg')
    
    embedding_avg = embedding_sum / len(tokens_processed)
    print(f'out of len(token_arr)={len(token_arr)} len(tokens_processed)={len(tokens_processed)} len(tokens_failed)={len(tokens_failed)}')
    return embedding_avg.tolist(), tokens_processed, tokens_failed


def generate_openai_embeddings(input_text: str, model_version: str) -> tuple[list, int, int]:
    print(f'begin generate_openai_embeddings for model_version={model_version}')

    openai_client = OpenAI(api_key=settings.openai_api_key)
    embeddings = []
    try:
        embeddings_response = openai_client.embeddings.create(model=model_version, input=input_text, encoding_format="float")
        # print(f'embeddings_response={embeddings_response}')
        if embeddings_response and embeddings_response.data and len(embeddings_response.data) > 0 and embeddings_response.data[0].embedding:
            embeddings = embeddings_response.data[0].embedding
            prompt_tokens_count = embeddings_response.usage.prompt_tokens
            total_tokens_count = embeddings_response.usage.total_tokens
            # TODO I'm not sure I understand what these fields and counts represent
            failed_tokens_count = total_tokens_count - prompt_tokens_count
            print(f'total_tokens_count={total_tokens_count} failed_tokens_count={failed_tokens_count}')
            return embeddings, total_tokens_count, failed_tokens_count
        else:
            print(f'Failed to generate openai:{model_version} vector embeddings: embeddings_response lacked data: {embeddings_response}')
            raise Exception(f'Failed to generate openai:{model_version} vector embeddings')
    except openai.BadRequestError as bre:
        print(f'Failed to generate openai:{model_version} vector embeddings: {bre}')
        raise bre
    except Exception as e:
        print(f'Failed to generate openai:{model_version} vector embeddings: {e}')
        raise Exception(f'Failed to generate openai:{model_version} vector embeddings: {e}', e)


@DeprecationWarning
# TODO refactor to use generate_embeddings
def generate_episode_embeddings(show_key: str, es_episode: EsEpisodeTranscript, model_vendor: str, model_version: str) -> None|Exception:
    print(f'begin generate_episode_embeddings for {show_key}:{es_episode.episode_key} using model {model_vendor}:{model_version}')

    if model_vendor == 'openai':
        vendor_meta = TRF_MODELS[model_vendor]
        true_model_version = vendor_meta['versions'][model_version]['true_name']
        try:
            embeddings, tokens, no_match_tokens = generate_openai_embeddings(es_episode.flattened_text, true_model_version)
            es_episode[f'{model_vendor}_{model_version}_embeddings'] = embeddings
        except openai.BadRequestError as bre:
            print(f'Failed to generate openai:{model_version} vector embeddings: {bre}')
            # If BadRequestError is about token count, iteratively retry request with slightly smaller variation of content until request goes thru 
            if "This model's maximum context length is 8192 tokens" in bre.message:
                skip_increment = 8
                success = False
                while not success and skip_increment > 1:
                    try:
                        embeddings, tokens, no_match_tokens = generate_openai_embeddings(shorten_flattened_text(es_episode, skip_increment=skip_increment), true_model_version)
                        es_episode[f'{model_vendor}_{model_version}_embeddings'] = embeddings
                        success = True
                    except Exception as e:
                        print(f'On retry using shorterned content, still failed to generate {model_vendor}:{model_version} vector embeddings for {show_key}:{es_episode.episode_key}: {e}')
                        skip_increment -= 1
                if not success:
                    raise Exception(f'On multiple retries using incrementally shorterned content, still failed to generate {model_vendor}:{model_version} vector embeddings for {show_key}:{es_episode.episode_key}: {e}')
        except Exception as e:
            print(f'Failed to generate {model_vendor}:{model_version} vector embeddings for {show_key}:{es_episode.episode_key}: {e}')
            raise Exception(f'Failed to generate {model_vendor}:{model_version} vector embeddings for {show_key}:{es_episode.episode_key}: {e}')

    else:
        vendor_meta = W2V_MODELS[model_vendor]
        tag_pos = vendor_meta['pos_tag']

        doc_tokens = []
        doc_tokens.extend(qpp.tokenize_and_remove_stopwords(es_episode.title, tag_pos=tag_pos))
        for scene in es_episode.scenes:
            scene_tokens = []
            # scene_tokens.extend(standardize_and_tokenize(scene.location, tag_pos=tag_pos))
            if scene.description:
                scene_tokens.extend(qpp.tokenize_and_remove_stopwords(scene.description, tag_pos=tag_pos))
            for scene_event in scene.scene_events:
                if scene_event.context_info:
                    scene_tokens.extend(qpp.tokenize_and_remove_stopwords(scene_event.context_info, tag_pos=tag_pos))
                # if scene_event.spoken_by:
                #     scene_tokens.extend(standardize_and_tokenize(scene_event.spoken_by, tag_pos=tag_pos))
                if scene_event.dialog:
                    scene_tokens.extend(qpp.tokenize_and_remove_stopwords(scene_event.dialog, tag_pos=tag_pos))

            if len(scene_tokens) > 0:
                doc_tokens.extend(scene_tokens)

        print(f'+++++++++++ len(doc_tokens)={len(doc_tokens)}')

        if len(doc_tokens) > 0:
            try:
                embeddings, tokens, no_match_tokens = calculate_embeddings(doc_tokens, model_vendor, model_version)
            except Exception as e:
                raise Exception(f'Failed to generate {model_vendor}:{model_version} vector embeddings for {show_key}:{es_episode.episode_key}: {e}')
            es_episode[f'{model_vendor}_{model_version}_embeddings'] = embeddings
            es_episode[f'{model_vendor}_{model_version}_tokens'] = tokens
            es_episode[f'{model_vendor}_{model_version}_no_match_tokens'] = no_match_tokens
        

# TODO incorporate Word2Vec embeddings generation from generate_episode_embeddings into this generic function
def generate_embeddings(text_to_vectorize: str, model_vendor: str, model_version: str) -> list|Exception:
    tokens = text_to_vectorize.split(' ')
    openai_token_count = openai_token_counter(text_to_vectorize)
    print(f'begin generate_embeddings text_to_vectorize len(tokens)={len(tokens)} openai_token_count={openai_token_count} model {model_vendor}:{model_version}')
    
    if model_vendor == 'openai':
        vendor_meta = TRF_MODELS[model_vendor]
        true_model_version = vendor_meta['versions'][model_version]['true_name']
        try:
            embeddings, _, _ = generate_openai_embeddings(text_to_vectorize, true_model_version)
            return embeddings
        except openai.BadRequestError as bre:
            # TODO slice up request and retry
            print(f'Failed to generate {model_vendor}:{model_version} vector embeddings for len(tokens)={len(tokens)}: {bre}')
            raise Exception(f'Failed to generate {model_vendor}:{model_version} vector embeddings for len(tokens)={len(tokens)} in text_to_vectorize={text_to_vectorize}: {bre}')
        except Exception as e:
            print(f'Failed to generate {model_vendor}:{model_version} vector embeddings for len(tokens)={len(tokens)} in text_to_vectorize={text_to_vectorize}: {e}')
            raise Exception(f'Failed to generate {model_vendor}:{model_version} vector embeddings for len(tokens)={len(tokens)} in text_to_vectorize={text_to_vectorize}: {e}')
        

def build_embeddings_model(show_key: str) -> dict:
    print(f'begin build_embeddings_model for show_key={show_key}')
    
    training_fragments = []

    # fetch all episodes for show_key
    doc_ids_response = esr.fetch_doc_ids(ShowKey(show_key))
    for doc_id in doc_ids_response['doc_ids']:
        episode_key = doc_id.split('_')[1]
        print(f'begin compiling training_fragments for episode_key={episode_key}')
        es_episode = EsEpisodeTranscript.get(id=f'{show_key}_{episode_key}')
        training_fragments.append(qpp.tokenize_and_remove_stopwords(es_episode.title))
        for scene in es_episode.scenes:
            # entries.append(preprocess_text(scene.location))
            for scene_event in scene.scene_events:
                if scene_event.context_info:
                    training_fragments.append(qpp.tokenize_and_remove_stopwords(scene_event.context_info))
                # if scene_event.spoken_by:
                #     entries.append(preprocess_text(scene_event.spoken_by))
                if scene_event.dialog:
                    training_fragments.append(qpp.tokenize_and_remove_stopwords(scene_event.dialog))
        print(f'len(training_fragments)={len(training_fragments)}')

    cbow_model = Word2Vec(sentences=training_fragments, min_count=1, vector_size=100, window=5)
    cbow_model_file_path = f'./w2v_models/homegrown/cbow_{show_key}.model'
    cbow_model.save(cbow_model_file_path)

    sg_model = Word2Vec(sentences=training_fragments, min_count=1, vector_size=100, window=5, sg=1)
    sg_model_file_path = f'./w2v_models/homegrown/sg_{show_key}.model'
    sg_model.save(sg_model_file_path)

    response = {}
    response['cbow_file_path'] = cbow_model_file_path
    response['cbow_file_size'] = os.path.getsize(cbow_model_file_path)
    response['cbow_wv_count'] = len(cbow_model.wv)
    response['sg_file_path'] = sg_model_file_path
    response['sg_file_size'] = os.path.getsize(sg_model_file_path)
    response['sg_wv_count'] = len(sg_model.wv)

    return response


def cluster_docs(doc_embeddings: dict, num_clusters: int):
    print(f'begin cluster_docs for doc_embeddings num_clusters={num_clusters}')

    doc_clusters_df = pd.DataFrame(doc_embeddings)
    doc_clusters_df = doc_clusters_df.transpose()
    # doc_clusters_df.index.name = 'doc_id' # TODO is this needed or do I create a separate column and have index be an incrementing int 0-n

    # doc_clusters = {}
    doc_ids = list(doc_embeddings.keys())
    embeddings = list(doc_embeddings.values())

    embeddings_matrix = np.vstack(embeddings)
    print(f'matrix.shape={embeddings_matrix.shape}')
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    kmeans.fit(embeddings_matrix)

    print(f'kmeans.labels_={kmeans.labels_} len(kmeans.labels_)={len(kmeans.labels_)} type(kmeans.labels_)={type(kmeans.labels_)}')

    # labels = kmeans.labels_.tolist()
    doc_clusters_df['cluster'] = kmeans.labels_
    doc_clusters_df['doc_id'] = doc_ids # TODO redundancy here with index name

    doc_clusters_df.columns = doc_clusters_df.columns.astype(str)

    # for i in range(len(doc_ids)):
    #     doc_clusters[doc_ids[i]] = labels[i]

    # return doc_clusters, doc_clusters_df, embeddings_matrix
    return doc_clusters_df


@DeprecationWarning
# TODO `generate_episode_embeddings` is only dependency and it is Deprecated, refactor everything to use generate_embeddings
def shorten_flattened_text(es_episode: EsEpisodeTranscript, skip_increment: int = None) -> str:
    flattened_text = f'{es_episode.title} '
    scene_i = 0
    for scene in es_episode.scenes:
        scene_i += 1
        # if divisor is set, skip scenes at that skip_increment
        if skip_increment and scene_i % skip_increment == 0:
            continue
        # flattened_text += f'{scene.location} '
        # if scene.description:
        #     flattened_text += f'{scene.description} '
        for scene_event in scene.scene_events:
            # if scene_event.context_info:
            #     flattened_text += f'{scene_event.context_info} '
            # if scene_event.spoken_by:
            #     flattened_text += f'{scene_event.spoken_by}: '
            if scene_event.dialog:
                flattened_text += f'{scene_event.dialog} '

    return flattened_text


def shorten_lines_of_text(lines: list, target_token_count: int) -> str:
    '''
    Slim down text that is (slightly) longer than openai embeddings endpoint can handle by zapping lines incrementally distributed throughout the text, 
    rather than lopping off large sections at the beginning or end (where the meaning might be most impactful)
    '''
    openai_token_count = openai_token_counter(' '.join(lines), 'cl100k_base')
    skip_increment = math.ceil(target_token_count / (openai_token_count - target_token_count))
    print(f'In shorten_lines_of_text len(lines)={len(lines)} openai_token_count={openai_token_count} target_token_count={target_token_count} skip_increment={skip_increment}')
    
    success = False
    while not success and skip_increment > 1:
        lines_out = []
        for i in range(len(lines)):
            if i % skip_increment == skip_increment-1:
                continue
            lines_out.append(lines[i])
        openai_token_count = openai_token_counter(' '.join(lines_out), 'cl100k_base')
        if openai_token_count < target_token_count:
            success = True
        else:
            print(f"skip_increment={skip_increment} only got us down to openai_token_count={openai_token_count}, reducing increment by 1")
            skip_increment -= 1
    if not success:
        print(f'Cannot run shorten_lines_of_text for skip_increment={skip_increment}')
        return None
    
    return lines_out


def openai_token_counter(raw_text: str, openai_encoding_version: str = None) -> int:
    if not openai_encoding_version:
        # openai_encoding_version = 'gpt-3.5-turbo'
        openai_encoding_version = 'cl100k_base'
    enc = tiktoken.get_encoding(openai_encoding_version)
    return len(enc.encode(raw_text))
