import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts
import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import re
import string
import warnings

from es_model import EsEpisodeTranscript, EsScene, EsSceneEvent
import main
from show_metadata import ShowKey


warnings.filterwarnings(action = 'ignore')


nltk.download('stopwords') # run this command to download the stopwords in the project
nltk.download('punkt') # essential for tokenization


loaded_models = {}


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
    model_path = f'./{vendor}/{version}/model.txt'
    if model_path in loaded_models:
        print(f'found model_path={model_path} in previously loaded models')
        word_vectors = loaded_models[model_path]
    else:
        print(f'did not find model_path={model_path} in previously loaded models, loading now...')
        word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)
        loaded_models[model_path] = word_vectors
        print(f'model_path={model_path} len(word_vectors)={len(word_vectors)} type(word_vectors)={type(word_vectors)}')
    # print(word_vectors.most_similar("vacation_NOUN"))
    # print(word_vectors.most_similar(positive=['woman_NOUN', 'king_NOUN'], negative=['man_NOUN']))

    return word_vectors


def preprocess_text(text: str, tag_pos: bool = False) -> str:
    # remove numbers and special characters
    text = re.sub("[^A-Za-z]+", " ", text)
    # tokenize
    tokens = word_tokenize(text)
    # remove stopwords
    tokens = [w.lower().strip() for w in tokens if not w.lower() in stopwords.words("english")]
    # pos tag
    if tag_pos:
        pos_tokens = pos_tag(tokens, tagset='universal')
        for i in range(len(pos_tokens)):
            tokens[i] = f'{pos_tokens[i][0]}_{pos_tokens[i][1]}'

    print(f'tokens={tokens}')
    return tokens


def calculate_embedding(token_arr: list, model_vendor: str, model_version: str) -> (list, list, list):
    print(f'begin calculate_embedding for token_arr={token_arr} model_vendor={model_vendor} model_version={model_version}')

    keyed_vectors = load_keyed_vectors(model_vendor, model_version)

    embedding_sum = [0.0] * 300
    tokens_processed = []
    tokens_failed = []
    for token in token_arr:
        if token not in keyed_vectors.key_to_index:
            print(f'did not find token={token}, skipping')
            tokens_failed.append(token)
        else:
            embedding_sum += keyed_vectors.get_vector(token)
            tokens_processed.append(token)
    
    if len(tokens_processed) == 0:
        raise Exception('No tokens processed, cannot calculate embedding avg')
    
    embedding_avg = embedding_sum / len(tokens_processed)
    print(f'out of len(token_arr)={len(token_arr)} len(tokens_processed)={len(tokens_processed)} len(tokens_failed)={len(tokens_failed)} embedding_avg={embedding_avg} ')
    return embedding_avg.tolist(), tokens_processed, tokens_failed


def generate_episode_embeddings(show_key: str, es_episode: EsEpisodeTranscript) -> None|Exception:
    print(f'begin generate_embeddings for show_key={show_key} episode_key={es_episode.episode_key}')

    # cbow_model = await load_model(show_key, 'cbow')
    # sg_model = await load_model(show_key, 'sg')

    doc_tokens = []
    doc_tokens.extend(preprocess_text(es_episode.title, tag_pos=True))
    for scene in es_episode.scenes:
        scene_tokens = []
        # scene_tokens.extend(preprocess_text(scene.location, tag_pos=True))
        if scene.description:
            scene_tokens.extend(preprocess_text(scene.description, tag_pos=True))
        for scene_event in scene.scene_events:
            if scene_event.context_info:
                scene_tokens.extend(preprocess_text(scene_event.context_info, tag_pos=True))
            # if scene_event.spoken_by:
            #     scene_tokens.extend(preprocess_text(scene_event.spoken_by, tag_pos=True))
            if scene_event.dialog:
                scene_tokens.extend(preprocess_text(scene_event.dialog, tag_pos=True))
            
        if len(scene_tokens) > 0:
            doc_tokens.extend(scene_tokens)

    if len(doc_tokens) > 0:
        # es_episode.cbow_doc_embedding = await calculate_embedding(doc_tokens, cbow_model)
        # es_episode.skipgram_doc_embedding = await calculate_embedding(doc_tokens, sg_model)
        try:
            webvec_29_embeddings, webvec_29_tokens, webvec_29_no_match_tokens = calculate_embedding(doc_tokens, 'webvectors', '29')
            webvec_223_embeddings, webvec_223_tokens, webvec_223_no_match_tokens = calculate_embedding(doc_tokens, 'webvectors', '223')
        except Exception as e:
            raise Exception(f'Failed to generate vector embeddings for show_key={show_key} episode_key={es_episode.episode_key}: {e}')
        es_episode.webvectors_29_embeddings = webvec_29_embeddings
        es_episode.webvectors_223_embeddings = webvec_223_embeddings
        es_episode.webvectors_29_tokens = webvec_29_tokens
        es_episode.webvectors_223_tokens = webvec_223_tokens
        es_episode.webvectors_29_no_match_tokens = webvec_29_no_match_tokens
        es_episode.webvectors_223_no_match_tokens = webvec_223_no_match_tokens


'''
Not being used so dependencies are out of date
'''
async def build_embeddings_model(show_key: str) -> dict:
    print(f'begin build_embeddings_model for show_key={show_key}')
    
    entries = []

    # fetch all episodes for show_key
    doc_ids_response = await main.search_doc_ids(ShowKey(show_key))
    for doc_id in doc_ids_response['doc_ids']:
        episode_key = doc_id.split('_')[1]
        print(f'begin compiling entries for episode_key={episode_key}')
        es_episode_response = await main.fetch_episode(ShowKey(show_key), episode_key, data_source='es')
        es_episode = es_episode_response['es_episode']
        entries.append(preprocess_text(es_episode['title']))
        if 'scenes' not in es_episode:
            continue
        for scene in es_episode['scenes']:
            # entries.append(preprocess_text(scene['location']))
            if 'scene_events' not in scene:
                continue
            for scene_event in scene['scene_events']:
                if 'context_info' in scene_event:
                    entries.append(preprocess_text(scene_event['context_info']))
                # if 'spoken_by' in scene_event:
                #     entries.append(preprocess_text(scene_event['spoken_by']))
                if 'dialog' in scene_event:
                    entries.append(preprocess_text(scene_event['dialog']))
        print(f'len(entries)={len(entries)}')

    cbow_model = Word2Vec(sentences=entries, min_count = 1, vector_size = 100, window = 5)
    cbow_model_file_name = f'w2v_cbow_{show_key}.model'
    cbow_model.save(cbow_model_file_name)

    sg_model = Word2Vec(sentences=entries, min_count = 1, vector_size = 100, window = 5, sg = 1)
    sg_model_file_name = f'w2v_sg_{show_key}.model'
    sg_model.save(sg_model_file_name)

    response = {}
    response['cbow_file_size'] = os.path.getsize(cbow_model_file_name)
    response['cbow_wv_count'] = len(cbow_model.wv)
    response['sg_file_size'] = os.path.getsize(sg_model_file_name)
    response['sg_wv_count'] = len(sg_model.wv)

    return response



    # model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
    # model.save("word2vec.model")
    # model = Word2Vec.load("word2vec.model")
    # model.train([["hello", "world"]], total_examples=1, epochs=1)
    # words = list(model.wv.index_to_key)


# # iterate through each sentence in the file
# for i in sent_tokenize(f):
#     temp = []
     
#     # tokenize the sentence into words
#     for j in word_tokenize(i):
#         temp.append(j.lower())
 
#     data.append(temp)
 
# # Create CBOW model
# model1 = Word2Vec(data, min_count = 1, vector_size = 100, window = 5)
 
# # Print results
# print("Cosine similarity between 'alice' " +
#                "and 'wonderland' - CBOW : ",
#     model1.wv.similarity('alice', 'wonderland'))
     
# print("Cosine similarity between 'alice' " +
#                  "and 'machines' - CBOW : ",
#       model1.wv.similarity('alice', 'machines'))
 
# # Create Skip Gram model
# model2 = Word2Vec(data, min_count = 1, vector_size = 100, window = 5, sg = 1)
 
# # Print results
# print("Cosine similarity between 'alice' " +
#           "and 'wonderland' - Skip Gram : ",
#     model2.wv.similarity('alice', 'wonderland'))
     
# print("Cosine similarity between 'alice' " +
#             "and 'machines' - Skip Gram : ",
#       model2.wv.similarity('alice', 'machines'))
    