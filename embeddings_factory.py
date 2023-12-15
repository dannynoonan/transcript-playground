import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import nltk
from nltk.corpus import stopwords
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


def preprocess_text(text: str) -> str:
    # remove numbers and special characters
    text = re.sub("[^A-Za-z]+", " ", text)
    # create tokens
    tokens = nltk.word_tokenize(text)
    # check if it's a stopword
    tokens = [w.lower().strip() for w in tokens if not w.lower() in stopwords.words("english")]
    # return a list of cleaned tokens
    return tokens


async def load_model(show_key: str, model_type: str) -> Word2Vec:
    model_file_name = f'w2v_{model_type}_{show_key}.model'
    model = Word2Vec.load(model_file_name)
    return model


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


async def generate_episode_embeddings(show_key: str, es_episode: EsEpisodeTranscript) -> None:
    print(f'begin generate_embeddings for show_key={show_key} episode_key={es_episode.episode_key}')

    cbow_model = await load_model(show_key, 'cbow')
    sg_model = await load_model(show_key, 'sg')

    doc_tokens = []
    doc_tokens.extend(preprocess_text(es_episode.title))
    for scene in es_episode.scenes:
        scene_tokens = []
        # scene_tokens.extend(preprocess_text(scene.location))
        if scene.description:
            scene_tokens.extend(preprocess_text(scene.description))
        for scene_event in scene.scene_events:
            if scene_event.context_info:
                scene_tokens.extend(preprocess_text(scene_event.context_info))
            # if scene_event.spoken_by:
            #     scene_tokens.extend(preprocess_text(scene_event.spoken_by))
            if scene_event.dialog:
                scene_tokens.extend(preprocess_text(scene_event.dialog))
            
        if len(scene_tokens) > 0:
            doc_tokens.extend(scene_tokens)

    if len(doc_tokens) > 0:
        es_episode.cbow_doc_embedding = await calculate_embedding(doc_tokens, cbow_model)
        es_episode.skipgram_doc_embedding = await calculate_embedding(doc_tokens, sg_model)

    # return es_episode


async def calculate_embedding(token_arr: list, model: Word2Vec) -> list:
    print(f'begin calculate_embedding for token_arr={token_arr} model={model}')
    embedding_sum = 0.0
    for token in token_arr:
        if token not in model.wv:
            print(f'while processing token_arr={token_arr} did not find token={token} in model={model}')
            continue
        embedding_sum += model.wv[token]
    embedding_avg = embedding_sum / len(token_arr)
    return embedding_avg.tolist()



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
    