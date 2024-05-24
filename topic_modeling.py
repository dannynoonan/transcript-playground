from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
# from huggingface_hub import login
import math
import os
import openai
from openai import OpenAI
import pandas as pd
import pickle
# import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

from app.config import settings
import app.es.es_read_router as esr
from app.nlp.nlp_metadata import MIN_WORDS_FOR_BERT, MAX_WORDS_FOR_BERT
from app.show_metadata import ShowKey


openai_client = OpenAI(api_key=settings.openai_api_key)
# openai.api_key = settings.openai_api_key

# embed_minilm = SentenceTransformer('all-MiniLM-L12-v2')



def main():
    print('begin topic_modeling')

    # print(f'attempt huggingface login')
    # login()
    # print(f'huggingface login success')

    show_key = 'TNG'


    bert_text_inputs = []
    bert_text_sources = []
    # small_deleted_scene_count = 0

    doc_ids_response = esr.fetch_doc_ids(ShowKey(show_key))
    for doc_id in doc_ids_response['doc_ids']:
    # for doc_id in ['TNG_238', 'TNG_218']:
        episode_key = doc_id.split('_')[1]
        
        # fetch narrative sequences for episode
        narrative_sequences_response = esr.fetch_narrative_sequences(ShowKey(show_key), episode_key)
        if 'narrative_sequences' not in narrative_sequences_response:
            print(f'Failure to fetch narrative_sequences for show_key={show_key} episode_key={episode_key}, skipping')
            continue
        narrative_sequences = narrative_sequences_response['narrative_sequences']

        # establish word counts per scene already sourced into narrative sequences, to avoid duplicating that dialog text during rote scene addition further down
        narrative_source_scene_wcs = {}
        for narr_seq in narrative_sequences:
            # convert source_scene_wcs keys back to ints for scene index comparison
            source_scene_wcs = {int(k):v for k,v in narr_seq['source_scene_word_counts'].items()}
            for ss, wc in source_scene_wcs.items():
                if ss not in narrative_source_scene_wcs:
                    narrative_source_scene_wcs[ss] = 0
                narrative_source_scene_wcs[ss] += wc

        # carve up and flatten narrative_lines into single narrative_text strings of bert-friendly token lengths
        bert_ready_narrative_sequences = []
        for narr_seq in narrative_sequences:
            if narr_seq['word_count'] < MAX_WORDS_FOR_BERT:
                narr_seq['narrative_text'] = ' '.join(narr_seq['narrative_lines'])
                # NOTE at this point narr_seq has a bunch of data it doesn't need, but also doesn't need to get rid of
                bert_ready_narrative_sequences.append(narr_seq)
            else:
                wc = 0
                narrative_subseq = []
                for i, line in enumerate(narr_seq['narrative_lines']):
                    line_wc = len(line.split(' '))
                    if (wc + line_wc) > MAX_WORDS_FOR_BERT:
                        narrative_text = ' '.join(narrative_subseq)
                        narr_subseq = dict(speaker_group=narr_seq['speaker_group'], narrative_text=narrative_text, word_count=wc)
                        bert_ready_narrative_sequences.append(narr_subseq)
                        narrative_subseq = []
                        wc = 0
                    narrative_subseq.append(line)
                    wc += line_wc
                    if i == len(narr_seq['narrative_lines'])-1:
                        narrative_text = ' '.join(narrative_subseq)
                        if len(narrative_text.split(' ')) > MIN_WORDS_FOR_BERT:
                            narr_subseq = dict(speaker_group=narr_seq['speaker_group'], narrative_text=narrative_text, word_count=wc)
                            bert_ready_narrative_sequences.append(narr_subseq)

        print('----------------------------------------------------------------------------------------')
        print(f'len(bert_ready_narrative_sequences)={len(bert_ready_narrative_sequences)} all_sourced_scene_wcs={narrative_source_scene_wcs} for episode_key={episode_key}')
        print(f'bert_ready_narrative_sequences={bert_ready_narrative_sequences}')

        # fetch all flattened scenes for episode, to cover portion of episode not sourced into narrative sequences
        flattened_scenes_response = esr.fetch_flattened_scenes(ShowKey(show_key), episode_key, include_speakers=False, include_context=True)
        episode_flattened_scenes = flattened_scenes_response['flattened_scenes']
        print(f'len(episode_flattened_scenes)={len(episode_flattened_scenes)} for episode_key={episode_key}')

        procd_episode_flattened_scenes = []
        for i in range(len(episode_flattened_scenes)):
            scene = episode_flattened_scenes[i]
            scene_wc = len(scene.split(' '))
            if scene_wc < MIN_WORDS_FOR_BERT:
                # if i in all_sourced_scene_wcs:
                #     print(f'episode_key={episode_key} scene={i} survived via narrative inclusion, otherwise it was too short and would have been dropped')
                # else:
                #     small_deleted_scene_count += 1
                continue
            if i in narrative_source_scene_wcs:
                # sourced_wc = narrative_source_scene_wcs[i]
                if narrative_source_scene_wcs[i] / scene_wc > 0.75:
                    print(f'episode_key={episode_key} scene={i} was {round(narrative_source_scene_wcs[i] / scene_wc, 2) * 100}% covered via narrative inclusion, so the standalone scene has been dropped')
                    continue
            if scene_wc < MAX_WORDS_FOR_BERT:
                procd_episode_flattened_scenes.append(scene)
            else:
                split_ct = math.ceil(scene_wc / MAX_WORDS_FOR_BERT)
                split_size = int(len(scene)/split_ct)
                for i in range(split_ct):
                    low = i * split_size
                    high = (i+1) * split_size
                    if high > scene_wc:
                        high = scene_wc
                    scene_segment = scene[low:high]
                    procd_episode_flattened_scenes.append(scene_segment)

        print('----------------------------------------------------------------------------------------')        
        print(f'len(procd_episode_flattened_scenes)={len(procd_episode_flattened_scenes)} for episode_key={episode_key}')

        bert_text_inputs.extend([ns['narrative_text'] for ns in bert_ready_narrative_sequences])
        bert_text_sources.extend([(f"{episode_key}_{ns['speaker_group']}", ns['word_count']) for ns in bert_ready_narrative_sequences])
        bert_text_inputs.extend(procd_episode_flattened_scenes)
        bert_text_sources.extend([(episode_key, len(scene.split(' '))) for scene in procd_episode_flattened_scenes])

        print(f'len(bert_text_inputs)={len(bert_text_inputs)} len(bert_text_sources)={len(bert_text_sources)} after episode_key={episode_key}')


    print('----------------------------------------------------------------------------------------')        
    print(f'len(bert_text_inputs)={len(bert_text_inputs)} len(bert_text_sources)={len(bert_text_sources)}')


    sources_df = pd.DataFrame(bert_text_sources, columns=['source', 'wc'])

    # all_scene_embeddings = []
    # for scene in all_flattened_scenes:
    #     scene_embeddings = embed_minilm.encode(scene)
    #     all_scene_embeddings.append(scene_embeddings)


    # start basic model
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    model_basic = BERTopic(vectorizer_model=vectorizer_model, language='english', calculate_probabilities=True, verbose=True)
    topics_basic, probs_basic = model_basic.fit_transform(bert_text_inputs)

    # model.push_to_hf_hub(repo_id='andyshirey/test_bert', save_ctfidf=True, save_embedding_model='sentence-transformers/all-MiniLM-L6-v2', serialization='pytorch')

    topic_freq_basic = model_basic.get_topic_info()
    print(f'topic_freq_basic={topic_freq_basic}')
    topic_freq_basic.to_csv('topic_data/topic_freq_basic.csv')

    doc_info_basic = model_basic.get_document_info(bert_text_inputs)
    print(f'doc_info_basic={doc_info_basic}')
    # df1 = pd.merge(doc_info_basic, sources_df, left_index=True, right_index=True)
    # df2 = doc_info_basic.join(sources_df)
    doc_info_basic = pd.concat([doc_info_basic, sources_df], axis=1)
    doc_info_basic['episode_key'] = doc_info_basic['source'].apply(lambda x: x.split('_')[0])
    doc_info_basic['prob_x_wc'] = doc_info_basic['Probability'] * doc_info_basic['wc']
    doc_info_basic.to_csv('topic_data/doc_info_basic.csv')

    # drop topic -1
    doc_info_basic = doc_info_basic[doc_info_basic['Topic'] > -1]
    # assemble tuple lists mapping episodes to topics
    topic_episode_rows = []
    for i in range(doc_info_basic['Topic'].max()):
        temp_df = doc_info_basic[doc_info_basic['Topic'] == i]
        e_keys = list(temp_df['episode_key'].unique())
        for e_key in e_keys:
            e_key_df = temp_df[temp_df.episode_key == e_key]
            topic_episode_rows.append(dict(topic=i, episode_key=e_key, mapping_cnt=len(e_key_df), high_prob=e_key_df['Probability'].max(), sum_weighted_probs=e_key_df['prob_x_wc'].sum())) 

    topic_episode_df = pd.DataFrame(topic_episode_rows)
    topic_episode_df = topic_episode_df.sort_values(['topic', 'sum_weighted_probs'], ascending=[True, False])
    topic_episode_df.to_csv('topic_data/topic_episode_basic.csv')

    model_basic.save('topic_models/model_basic')

    # scatter_fig = model.visualize_topics()
    # hierarchy_fig = model.visualize_hierarchy()


    # start custom model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    umap_model = UMAP(n_neighbors=3, n_components=3, min_dist=0.05)
    hdbscan_model = HDBSCAN(min_cluster_size=50, min_samples=30, gen_min_span_tree=True, prediction_data=True)

    model_custom = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
                            top_n_words=5, language='english', calculate_probabilities=True, verbose=True)
    topics_custom, probs_custom = model_custom.fit_transform(bert_text_inputs)

    topic_freq_custom = model_custom.get_topic_info()
    print(f'topic_freq_custom={topic_freq_custom}')
    topic_freq_custom.to_csv('topic_data/topic_freq_custom.csv')

    doc_info_custom = model_custom.get_document_info(bert_text_inputs)
    print(f'doc_info_custom={doc_info_custom}')
    doc_info_custom = pd.concat([doc_info_custom, sources_df], axis=1)
    doc_info_custom['episode_key'] = doc_info_custom['source'].apply(lambda x: x.split('_')[0])
    doc_info_custom.to_csv('topic_data/doc_info_custom.csv')

    # model_custom.push_to_hf_hub(repo_id='andyshirey/test_bert_custom', save_ctfidf=True, save_embedding_model=embed_model, serialization='pytorch')
    model_custom.save('topic_models/model_custom')

    # model_topic_info = model_custom.get_topic_info(topic=2)

    # bar_fig = model_custom.visualize_barchart()

    # flattened_scene_embeddings = embedding_model.encode(all_flattened_scenes, batch_size=64)
    # flattened_scene_umap = umap_model.transform(flattened_scene_embeddings)

    # fig = px.scatter_3d(docs_data[docs_data.Topic.isin([0,1,2]),x='x_coord',y='y_coord',z='z_coord',color='Topic',opacity=0.7,size='point_size')
    # fig.update_traces(marker=dict(line=dict(width=0.1, color='DarkSlateGrey')), selector=dict(mode='markers'))
                        

    # start repKb model
    representation_model = KeyBERTInspired()

    model_repKb = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
                           representation_model=representation_model, top_n_words=5, language='english', calculate_probabilities=True, verbose=True)
    topics_repKb, probs_repKb = model_repKb.fit_transform(bert_text_inputs)

    topic_freq_repKb = model_repKb.get_topic_info()
    print(f'topic_freq_repKb={topic_freq_repKb}')
    topic_freq_repKb.to_csv('topic_data/topic_freq_repKb.csv')

    doc_info_repKb = model_repKb.get_document_info(bert_text_inputs)
    print(f'doc_info_repKb={doc_info_repKb}')
    doc_info_repKb = pd.concat([doc_info_repKb, sources_df], axis=1)
    doc_info_repKb['episode_key'] = doc_info_repKb['source'].apply(lambda x: x.split('_')[0])
    doc_info_repKb.to_csv('topic_data/doc_info_repKb.csv')

    # model_repKb.push_to_hf_hub(repo_id='andyshirey/test_bert_repKb', save_ctfidf=True, save_embedding_model=embed_model, serialization='pytorch')
    model_repKb.save('topic_models/model_repKb')




def original_main():
    print('begin original_main')

    show_key = 'TNG'

    # docs = []
    # for i in range(1,8):
    #     response = esr.fetch_flattened_episodes(ShowKey('TNG'), i)
    #     for episode in response['episodes']:
    #         docs.append(episode['flattened_text'])

    docs = []
    doc_ids_response = esr.fetch_doc_ids(ShowKey(show_key))
    for doc_id in doc_ids_response['doc_ids']:
        episode_key = doc_id.split('_')[1]
        flattened_scenes_response = esr.fetch_flattened_scenes(ShowKey(show_key), episode_key, include_speakers=True, include_context=True)
        flattened_scenes = flattened_scenes_response['flattened_scenes']
        print(f'len(flattened_scenes)={len(flattened_scenes)} for episode_key={episode_key}')
        docs.extend(flattened_scenes)

    print(f'len(docs)={len(docs)}')

    topic_model = BERTopic(embedding_model='all-MiniLM-L6-v2')
    topics, probs = topic_model.fit_transform(docs)

    df = topic_model.get_topic_info()
    df.to_csv('topic_model_flat_scenes_speakers_context.csv')

    topic_meta0 = topic_model.get_topic(0)
    print(f'topic_meta0={topic_meta0}')

    topic_meta1 = topic_model.get_topic(1)
    print(f'topic_meta1={topic_meta1}')

    print(f'type(topics)={type(topics)}')
    print(f'len(topics)={len(topics)} len(probs)={len(probs)}')
    print(f'topics={topics} probs={probs}')

    # prompt = 'TODO'
    # model_response = get_model_response(docs, prompt)

    # print(f'type(model_response)={type(model_response)}')

    
def get_model_response(messages, prompt):
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=messages,
    #     temperature=temperature, 
    #     max_tokens=max_tokens, 
    # )
    response = openai_client.completions.create(
        prompt=prompt,
        messages=messages,
        model='gpt-3.5-turbo-instruct',
        top_p = 0.5, 
        max_tokens = 1000,
        stream=True
    )

    return response.choices[0].message['content']


if __name__ == '__main__':
    main()
