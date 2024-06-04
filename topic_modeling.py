import argparse
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI as BertOpenAI
from hdbscan import HDBSCAN
from itertools import chain, product
# from huggingface_hub import login
import math
from operator import itemgetter
import os
import openai
# from openai import OpenAI
import pandas as pd
import pickle
# import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from app.config import settings
import app.es.es_read_router as esr
from app.nlp.nlp_metadata import MIN_WORDS_FOR_BERT, MAX_WORDS_FOR_BERT
from app.show_metadata import ShowKey


openai_client = openai.OpenAI(api_key=settings.openai_api_key)
# openai.api_key = settings.openai_api_key

# embed_minilm = SentenceTransformer('all-MiniLM-L12-v2')


CONFIG_OPTIONS = {
    # 'narrative_only': [True, False],
    # 'sentence_transformer_lm': ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2'],
    'sentence_transformer_lm': 'all-MiniLM-L6-v2',
    'vec_ngram_low': 1,
    'vec_ngram_high': [2, 3],
    # 'vec_ngram_high': 2,
    # 'bertopic_top_n_words': [3, 4, 5, 6, 7],
    'bertopic_top_n_words': 5,
    'umap_n_neighbors': [2, 5, 10, 20],
    # 'umap_n_components': [3],
    'umap_n_components': 3,
    'umap_min_dist': [0.0, 0.02, 0.05],
    # 'umap_metric': ['euclidian', 'manhattan', 'chebyshev', 'minkowski', 'cosine', 'correlation' 
    #                 'canberra', 'braycurtis', 'haversine', 'mahalanobis', 'wminkowski', 'seuclidian'],
    'umap_metric': 'cosine',
    # 'umap_random_state': [v for v in range(1, 6)],
    'umap_random_state': [4, 14, 43, 64],
    # 'umap_random_state': 14,
    'hdbscan_min_cluster_size': [25, 50, 75],
    'hdbscan_min_samples': 10,
    # 'mmr_diversity': [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    'mmr_diversity': 0.05,
    # 'representation_model_type': ['Custom', 'MaximalMarginalRelevance', 'KeyBERTInspired', 'BertOpenAI'],
    'representation_model_type': 'Custom',
}


def main():
    print('begin topic_modeling')
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    # parser.add_argument("--model_names", "-m", help="BERTopic model names", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    # model_names = args.model_names
    # model_names = model_names.split(',')

    # print(f'attempt huggingface login')
    # login()
    # print(f'huggingface login success')

    config_params = []
    config_param_value_counts = []
    for config_param, config_values in CONFIG_OPTIONS.items():
        if isinstance(config_values, list):
            config_params.append(config_param)
            config_param_value_counts.append(len(config_values))
    
    print('------------------------------------------------------------------------------------------')
    print(f'config_params={config_params}')
    print(f'config_param_value_counts={config_param_value_counts}')

    mx = []
    for vc in config_param_value_counts:
        mx.append([i for i in range(vc)])

    print('------------------------------------------------------------------------------------------')
    print(f'mx={mx}')

    config_combos = list(product(*mx))

    print('------------------------------------------------------------------------------------------')
    print(f'config_combos={config_combos}')

    configs = []
    for config_combo in config_combos:
        config = generate_config_instance(config_params, config_combo)
        configs.append(config)

    print('------------------------------------------------------------------------------------------')
    print(f'len(configs)={len(configs)}')

    bert_text_inputs, bert_text_sources = generate_bert_text_inputs(ShowKey(show_key), narrative_only=False)
    # doc_count = len(bert_text_inputs)

    simple_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
    episodes = simple_episodes_response['episodes']
    episodes_df = pd.DataFrame(episodes)
    sources_df = pd.DataFrame(bert_text_sources, columns=['episode_key', 'speaker_group', 'wc'])
    sources_df = pd.merge(sources_df, episodes_df, on='episode_key')

    for config in configs:
        run_and_log_bert_config(bert_text_inputs, config, sources_df)


def generate_config_instance(config_params: list, config_value_indexes: tuple) -> dict:
    config_instance = {}
    for config_param, config_val in CONFIG_OPTIONS.items():
        if isinstance(config_val, list):
            config_param_i = config_params.index(config_param)
            config_val_i = config_value_indexes[config_param_i]
            config_instance[config_param] = config_val[config_val_i]
        else:
            config_instance[config_param] = config_val
    return config_instance


def run_and_log_bert_config(bert_text_inputs: list, config: dict, sources_df: pd.DataFrame):
    logs = []
    topic_count_threshold = 6
    topic_ratio_threshold = 0.5
    corr_threshold = 0.2
    match_threshold = 4

    vec_ngram_low = config['vec_ngram_low']
    vec_ngram_high = config['vec_ngram_high']
    representation_model_type = config['representation_model_type']
    sentence_transformer_lm = config['sentence_transformer_lm']
    umap_n_neighbors = config['umap_n_neighbors']
    umap_n_components = config['umap_n_components']
    umap_min_dist = config['umap_min_dist']
    umap_metric = config['umap_metric']
    umap_random_state = config['umap_random_state']
    hdbscan_min_cluster_size = config['hdbscan_min_cluster_size']
    hdbscan_min_samples = config['hdbscan_min_samples']
    mmr_diversity = config['mmr_diversity']
    bertopic_top_n_words = config['bertopic_top_n_words']

    # configurable component models pt 1
    vectorizer_model = CountVectorizer(ngram_range=(vec_ngram_low, vec_ngram_high), stop_words='english')
        
    if representation_model_type:
        # configurable component models pt 2
        embedding_model = SentenceTransformer(sentence_transformer_lm)
        umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, min_dist=umap_min_dist, metric=umap_metric, random_state=umap_random_state)
        hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples, gen_min_span_tree=True, prediction_data=True)

        if representation_model_type == 'MaximalMarginalRelevance':
            mmr_representation_model = MaximalMarginalRelevance(diversity=mmr_diversity)
            bertopic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
                                      representation_model=mmr_representation_model, top_n_words=bertopic_top_n_words, language='english', 
                                      calculate_probabilities=True, verbose=True)
        elif representation_model_type == 'KeyBERTInspired':
            kb_representation_model = KeyBERTInspired()
            bertopic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
                                      representation_model=kb_representation_model, top_n_words=bertopic_top_n_words, language='english', 
                                      calculate_probabilities=True, verbose=True)
        elif representation_model_type == 'BertOpenAI':
            openai_representation_model = BertOpenAI(openai_client, delay_in_seconds=5, model='gpt-3.5-turbo-instruct', chat=True)
            bertopic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
                                      representation_model=openai_representation_model, top_n_words=bertopic_top_n_words, language='english', 
                                      calculate_probabilities=True, verbose=True)
        elif representation_model_type == 'Custom':
            bertopic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
                                      top_n_words=bertopic_top_n_words, language='english', calculate_probabilities=True, verbose=True)
        else:
            print(f'Cannot instantiate bertopic_model for invalid representation_model_type={representation_model_type}, exiting.')
            return
    else:
        bertopic_model = BERTopic(vectorizer_model=vectorizer_model, language='english', calculate_probabilities=True, verbose=True)

    topics_custom, probs_custom = bertopic_model.fit_transform(bert_text_inputs)

    # print('--------------------------- bertopic_model for: ----------------------------------')
    # print(f'- vectorizer ngram range=({vec_ngram_low},{vec_ngram_high})')
    # print(f'- umap n_neighbors={umap_n_neighbors}/n_components={umap_n_components}/min_dist={umap_min_dist}/metric={umap_metric}/random_state={umap_random_state}')
    # print(f'- hdbscan min_cluster_size={hdbscan_min_cluster_size}/min_samples={hdbscan_min_samples}')

    bertopic_docs_df = bertopic_model.get_document_info(bert_text_inputs)
    bertopic_topics_df = bertopic_model.get_topic_info()

    # REPORTING
    topic_count = len(bertopic_topics_df)
    doc_count_with_topic = bertopic_topics_df[bertopic_topics_df['Topic'] >= 0]['Count'].sum()
    topic_coverage_ratio = doc_count_with_topic / bertopic_topics_df['Count'].sum()
    if topic_count >= topic_count_threshold and topic_coverage_ratio > topic_ratio_threshold:
        logs.append(f'topic_count={topic_count}, topic_coverage_ratio={topic_coverage_ratio}')
    else:
        return

    '''
    bertopic_model.visualize_barchart()
    bertopic_model.visualize_topics()
    bertopic_model.visualize_hierarchy()
    '''

    # TODO I'm still unclear on why embeddings won't work first time when representation_model_type is not fed into BERTopic
    if representation_model_type:
        embeds = embedding_model.encode(bert_text_inputs, batch_size=64)
        umapd_embeds = umap_model.transform(embeds)

        # before dropping rows, map reduced-dimension coords from umapd_embeds
        bertopic_docs_df.loc[:,'x_coord'] = [coord[0] for coord in umapd_embeds]
        bertopic_docs_df.loc[:,'y_coord'] = [coord[1] for coord in umapd_embeds]
        bertopic_docs_df.loc[:,'z_coord'] = [coord[2] for coord in umapd_embeds]
        bertopic_docs_df.loc[:,'point_size'] = [abs(round(coord[0],0)) for coord in umapd_embeds]

    # merge with sources_df
    bertopic_docs_df = pd.concat([bertopic_docs_df, sources_df], axis=1)

    # now that source df has been merged into bertopic_docs_df, remove  
    # remove topic -1
    bertopic_docs_df = bertopic_docs_df[bertopic_docs_df['Topic'] >= 0]

    # remove non-narrative entries
    bertopic_docs_df = bertopic_docs_df[bertopic_docs_df['speaker_group'] != '']

    # set prob_x_wc
    bertopic_docs_df['prob_x_wc'] = bertopic_docs_df['Probability'] * bertopic_docs_df['wc']
    # TODO: set point_size to 'prob_x_wc' or 'Probability'?

    # one-hot encoding of topics and speakers
    # Topic: back up as topic_str, then run get_dummies against column with single topic value 
    bertopic_prefix = 'Topic_'
    bertopic_docs_df['topic_str'] = bertopic_docs_df['Topic'].apply(lambda x: str(x))
    bertopic_docs_df = pd.get_dummies(bertopic_docs_df, columns=['Topic'], dtype=int, drop_first=False)
    
    # topics_focused_tfidf and speaker_group: convert to list, then one-hot encode values from the resulting 'list' column 
    # topics_focused_tfidf
    focused_topic_prefix = 'foctopic_'
    bertopic_docs_df['topics_focused_tfidf_list'] = bertopic_docs_df['topics_focused_tfidf'].apply(lambda x: [t['topic_key'] for t in x[:3]])
    distinct_focused_topics = sorted(set(chain.from_iterable(bertopic_docs_df['topics_focused_tfidf_list'])))
    distinct_focused_topics_series = pd.Series(distinct_focused_topics, index=[f'{focused_topic_prefix}{s}' for s in distinct_focused_topics])
    bertopic_docs_df = bertopic_docs_df.join(bertopic_docs_df['topics_focused_tfidf_list'].apply(lambda x: distinct_focused_topics_series.isin(x)).astype(int))
    # speaker_group 
    speaker_prefix = 'spkr_'
    bertopic_docs_df['speaker_group_list'] = bertopic_docs_df['speaker_group'].apply(lambda x: x.split('_'))
    distinct_speakers = sorted(set(chain.from_iterable(bertopic_docs_df['speaker_group_list'])))
    distinct_speaker_series = pd.Series(distinct_speakers, index=[f'{speaker_prefix}{s}' for s in distinct_speakers])
    bertopic_docs_df = bertopic_docs_df.join(bertopic_docs_df['speaker_group_list'].apply(lambda x: distinct_speaker_series.isin(x)).astype(int))

    # trim one-hot encoded speaker columns down to a manageable size
    spkr_count = 30
    speaker_quantities = {}
    for col_name in bertopic_docs_df.columns:
        if speaker_prefix in col_name:
            speaker = col_name.split('_')[1]
            speaker_quantities[speaker] = bertopic_docs_df[col_name].sum()
            
    top_speakers = sorted(speaker_quantities.items(), key=itemgetter(1), reverse=True)
    top_speakers = top_speakers[:spkr_count]

    top_speaker_list = [s[0] for s in top_speakers]
    for col_name in bertopic_docs_df.columns:
        if speaker_prefix in col_name:
            speaker = col_name.split('_')[1]
            if speaker not in top_speaker_list:
                bertopic_docs_df.drop(col_name, axis=1, inplace=True)  

    # generate corr dfs
    bertopic_cols = [c for c in bertopic_docs_df.columns if bertopic_prefix in c]
    foctopic_cols = [c for c in bertopic_docs_df.columns if focused_topic_prefix in c]
    speaker_cols = [c for c in bertopic_docs_df.columns if speaker_prefix in c]

    # bertopic_cols + foctopic_cols
    corr_values_bertopic_x_foctopic = extract_corr_values(bertopic_docs_df, bertopic_cols, foctopic_cols, corr_threshold)
    if len(corr_values_bertopic_x_foctopic) >= match_threshold:
        logs.append(f'corr_values_bertopic_x_foctopic={corr_values_bertopic_x_foctopic}')
    corr_values_bertopic_x_speaker = extract_corr_values(bertopic_docs_df, bertopic_cols, speaker_cols, corr_threshold)
    if len(corr_values_bertopic_x_speaker) >= match_threshold:
        logs.append(f'corr_values_bertopic_x_speaker={corr_values_bertopic_x_speaker}')
    corr_values_foctopic_x_speaker = extract_corr_values(bertopic_docs_df, foctopic_cols, speaker_cols, corr_threshold)
    if len(corr_values_foctopic_x_speaker) >= match_threshold:
        logs.append(f'corr_values_foctopic_x_speaker={corr_values_foctopic_x_speaker}')

    print('-----------------------------------------------------------------------------------------')
    print(f'config={config}')
    print(f'logs={logs}')
    f = open("bertopic_config_test.txt", "a")  # append mode
    f.write('-----------------------------------------------------------------------------------------\n')
    f.write(f'config: {config}\n')
    for log in logs:
        f.write(f'*** {log}\n')
    f.close()


    '''
    custom_data = ['topic_str', 'title', 'season', 'sequence_in_season', 'focal_speakers', 'x_coord', 'y_coord', 'z_coord', 'topics_focused_tfidf_str']

    fig = px.scatter_3d(bertopic_docs_df, x='x_coord', y='y_coord', z='z_coord', 
                            custom_data=custom_data, color='topic_str', opacity=0.7,  
                            # labels={'Topic', 'Topic'}, color_discrete_map=color_discrete_map, category_orders=category_orders,
                            height=800, width=1000)

    fig.update_traces(marker=dict(line=dict(width=0.1, color='DarkSlateGrey')), selector=dict(mode='markers'))

    fig.update_traces(
            hovertemplate = "<br>".join([
                "Topics: %{customdata[0]}",
                "Episode: %{customdata[1]} (S%{customdata[2]}:E%{customdata[3]})",            
                "Focal speakers: %{customdata[4]}",
                "Focal topics: %{customdata[8]}",
                "x: %{customdata[5]}",
                "y: %{customdata[6]}",
                "z: %{customdata[7]}"
            ])
        )
    '''
                        
    '''
    if 'keybert' in model_names:
        # repKb model
        representation_model = KeyBERTInspired()
        model_repKb = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
                               representation_model=representation_model, top_n_words=5, language='english', calculate_probabilities=True, verbose=True)
        topics_repKb, probs_repKb = model_repKb.fit_transform(bert_text_inputs)
        log_and_save_topics(model_repKb, "repKb", bert_text_inputs, sources_df)

    if 'openai' in model_names:
        # openai model
        # representation_model = BertOpenAI(openai_client, model="gpt-3.5-turbo", chat=True)
        representation_model = BertOpenAI(openai_client, delay_in_seconds=5, model='gpt-3.5-turbo-instruct', chat=True)
        # model_openai = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
        #                         representation_model=representation_model, top_n_words=5, language='english', calculate_probabilities=True, verbose=True)
        model_openai = BERTopic(representation_model=representation_model)

        # model_repKb = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
        #                        representation_model=representation_model, top_n_words=5, language='english', calculate_probabilities=True, verbose=True)
        topics_openai, probs_openai = model_openai.fit_transform(bert_text_inputs)
        log_and_save_topics(model_openai, "openAI", bert_text_inputs, sources_df)
    '''


def extract_corr_values(bertopic_docs_df: pd.DataFrame, cols1: list, cols2: list, threshold: float) -> list:
    bertopic_docs_df_trimmed = bertopic_docs_df[cols1 + cols2]
    corr_df = bertopic_docs_df_trimmed.corr()
    for col_name in corr_df.columns:
        if col_name in cols1:
            corr_df.drop(col_name, axis=1, inplace=True)
    for i, _ in corr_df.iterrows():
        if i in cols2:
            corr_df.drop(i, inplace=True)

    corr_vals = [[row, col, corr_df[col][row]] for row in corr_df.index for col in corr_df if corr_df[col][row] > threshold]

    return corr_vals


def generate_bert_text_inputs(show_key: ShowKey, narrative_only: bool = False) -> tuple[list, list]:
    bert_text_inputs = []
    bert_text_sources = []

    doc_ids_response = esr.fetch_doc_ids(show_key)
    for doc_id in doc_ids_response['doc_ids']:
    # for doc_id in ['TNG_251', 'TNG_256']:
        episode_key = doc_id.split('_')[1]
        
        # fetch narrative sequences for episode
        narrative_sequences_response = esr.fetch_narrative_sequences(show_key, episode_key)
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

        # print(f'len(bert_ready_narrative_sequences)={len(bert_ready_narrative_sequences)} all_sourced_scene_wcs={narrative_source_scene_wcs} for episode_key={episode_key}')

        bert_text_inputs.extend([ns['narrative_text'] for ns in bert_ready_narrative_sequences])
        bert_text_sources.extend([(episode_key, ns['speaker_group'], ns['word_count']) for ns in bert_ready_narrative_sequences])

        if not narrative_only:

            # fetch all flattened scenes for episode, to cover portion of episode not sourced into narrative sequences
            flattened_scenes_response = esr.fetch_flattened_scenes(ShowKey(show_key), episode_key)
            episode_flattened_scenes = flattened_scenes_response['flattened_scenes']
            # print(f'len(episode_flattened_scenes)={len(episode_flattened_scenes)} for episode_key={episode_key}')

            procd_episode_flattened_scenes = []
            for i in range(len(episode_flattened_scenes)):
                scene = episode_flattened_scenes[i]
                scene_wc = len(scene.split(' '))
                if scene_wc < MIN_WORDS_FOR_BERT:
                    continue
                if i in narrative_source_scene_wcs:
                    if narrative_source_scene_wcs[i] / scene_wc > 0.75:
                        continue
                if scene_wc < MAX_WORDS_FOR_BERT:
                    procd_episode_flattened_scenes.append(scene)
                else:
                    split_ct = math.ceil(scene_wc / MAX_WORDS_FOR_BERT)
                    split_size = int(len(scene.split(' ')) / split_ct)
                    for i in range(split_ct):
                        low = i * split_size
                        high = (i+1) * split_size
                        if high > scene_wc:
                            high = scene_wc
                        scene_segment = scene[low:high]
                        procd_episode_flattened_scenes.append(scene_segment)

            # print(f'len(procd_episode_flattened_scenes)={len(procd_episode_flattened_scenes)} for episode_key={episode_key}')
            bert_text_inputs.extend(procd_episode_flattened_scenes)
            bert_text_sources.extend([(episode_key, '', len(scene.split(' '))) for scene in procd_episode_flattened_scenes])

    print(f'len(bert_text_inputs)={len(bert_text_inputs)} len(bert_text_sources)={len(bert_text_sources)}')
    return bert_text_inputs, bert_text_sources


# def log_topics(model: BERTopic, bert_text_inputs: list) -> tuple[pd.DataFrame, pd.DataFrame]:
#     doc_info_df = model.get_document_info(bert_text_inputs)
#     topic_freq_df = model.get_topic_info()
#     topic_count = len(topic_freq_df)
#     doc_count_with_topic = topic_freq_df[topic_freq_df['Topic'] >= 0]['Count'].sum()
#     topic_coverage_ratio = doc_count_with_topic / topic_freq_df['Count'].sum()
#     print(f'>>> topic_count={topic_count}, topic_coverage_ratio={topic_coverage_ratio}')

#     return doc_info_df, topic_freq_df


def log_and_save_topics(model: BERTopic, model_name: str, bert_text_inputs: list, sources_df: pd.DataFrame):
    print('----------------------------------------------------------------------------')
    print(f'BEGIN REPORT for model_name={model_name}')
    topic_freq = model.get_topic_info()
    print(f'topic_freq={topic_freq}')
    topic_freq.to_csv(f'topic_data/topic_freq_{model_name}.csv')

    for i in range(topic_freq['Topic'].max()):
        print(f'model.get_topic({i})={model.get_topic(i)}')

    doc_info = model.get_document_info(bert_text_inputs)
    print(f'doc_info={doc_info}')
    doc_info = pd.concat([doc_info, sources_df], axis=1)
    doc_info['episode_key'] = doc_info['source'].apply(lambda x: x.split('_')[0])
    doc_info['prob_x_wc'] = doc_info['Probability'] * doc_info['wc']
    doc_info.to_csv(f'topic_data/doc_info_{model_name}.csv')

    # drop topic -1
    doc_info = doc_info[doc_info['Topic'] > -1]
    # assemble tuple lists mapping episodes to topics
    topic_episode_rows = []
    for i in range(doc_info['Topic'].max()):
        temp_df = doc_info[doc_info['Topic'] == i]
        e_keys = list(temp_df['episode_key'].unique())
        for e_key in e_keys:
            e_key_df = temp_df[temp_df['episode_key'] == e_key]
            topic_episode_rows.append(dict(topic=i, episode_key=e_key, mapping_cnt=len(e_key_df), high_prob=e_key_df['Probability'].max(), sum_weighted_probs=e_key_df['prob_x_wc'].sum())) 

    topic_episode_df = pd.DataFrame(topic_episode_rows)
    topic_episode_df = topic_episode_df.sort_values(['topic', 'sum_weighted_probs'], ascending=[True, False])
    topic_episode_df.to_csv(f'topic_data/topic_episode_{model_name}.csv')

    # model.push_to_hf_hub(repo_id=f'andyshirey/test_bert_{model_name}', save_ctfidf=True, save_embedding_model=embed_model, serialization='pytorch')
    model.save(f'topic_models/model_{model_name}', serialization='safetensors')


def dumped():
    # bad_bert_input_indexes = []
    # for i in range(len(bert_text_inputs)):
    #     bti = bert_text_inputs[i]
    #     if len(bti) == 0:
    #         bad_bert_input_indexes.append(i)

    # print(f'bad_bert_input_indexes={bad_bert_input_indexes}')

    # for i in bad_bert_input_indexes:
    #     print(f'bert_text_sources[{i}]={bert_text_sources[i]}')

    # all_scene_embeddings = []
    # for scene in all_flattened_scenes:
    #     scene_embeddings = embed_minilm.encode(scene)
    #     all_scene_embeddings.append(scene_embeddings)

    # umap_model = UMAP(n_neighbors=3, n_components=3, min_dist=0.05) # used thru v7
    # umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42) # v8
    # umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=42) # v9
    # umap_model = UMAP(n_neighbors=20, n_components=5, min_dist=0.0, metric='cosine', random_state=42) # v10
    # umap_model = UMAP(n_neighbors=3, n_components=3, min_dist=0.0, metric='cosine', random_state=42) # v11


    # if 'basic' in model_names:
    #     # basic model
        
    #     model_basic = BERTopic(vectorizer_model=vectorizer_model, language='english', calculate_probabilities=True, verbose=True)
    #     topics_basic, probs_basic = model_basic.fit_transform(bert_text_inputs)
    #     print(f'model_basic for vectorizer ngram range=({vec_ngram_low},{vec_ngram_high})')
    #     doc_info_basic, topic_freq_basic = log_topics(model_basic, bert_text_inputs)

        # log_and_save_topics(model_basic, "basic", bert_text_inputs, sources_df)
        # word_score_fig = model_basic.visualize_barchart()
        # intertopic_dist_fig = model_basic.visualize_topics()
        # hierarchy_fig = model_basic.visualize_hierarchy()
    

    # one hot encoding of topics and speakers
    # doc_info_custom_agg_topics = doc_info_custom.groupby('episode_key', as_index=False)['Topic_str'].unique()
    # doc_info_custom['speaker_group_list'] = doc_info_custom['speaker_group'].apply(lambda x: x.split('_'))
    # doc_info_custom_agg_speakers = doc_info_custom.groupby('episode_key', as_index=False)['speaker_group_list'].unique()

    # doc_info_custom_agg_final = pd.merge(doc_info_custom_agg, doc_info_custom_agg_topics, on='episode_key')
    # doc_info_custom_agg_final = pd.merge(doc_info_custom_agg_final, sources_df, on='episode_key')

    return


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


LEGACY_CONFIG_OPTIONS_JUN03 = {
    # 'narrative_only': [True, False],
    # 'sentence_transformer_lm': ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2'],
    'sentence_transformer_lm': 'all-MiniLM-L6-v2',
    'vec_ngram_low': 1,
    # 'vec_ngram_high': [2, 3, 4],
    'vec_ngram_high': 2,
    # 'bertopic_top_n_words': [3, 4, 5, 6, 7],
    'bertopic_top_n_words': 5,
    'umap_n_neighbors': [2, 5, 10, 25, 50, 100, 200],
    # 'umap_n_components': [3],
    'umap_n_components': 3,
    'umap_min_dist': [0.0, 0.02, 0.05, 0.1, 0.3, 0.8],
    # 'umap_metric': ['euclidian', 'manhattan', 'chebyshev', 'minkowski', 'cosine', 'correlation' 
    #                 'canberra', 'braycurtis', 'haversine', 'mahalanobis', 'wminkowski', 'seuclidian'],
    'umap_metric': 'cosine',
    # 'umap_random_state': [v for v in range(1, 6)],
    # 'umap_random_state': [4, 14, 43, 64],
    'umap_random_state': 14,
    'hdbscan_min_cluster_size': [25, 50, 100],
    'hdbscan_min_samples': [10, 25, 50],
    # 'mmr_diversity': [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    'mmr_diversity': 0.05,
    # 'representation_model_type': ['Custom', 'MaximalMarginalRelevance', 'KeyBERTInspired', 'BertOpenAI'],
    'representation_model_type': 'Custom',
}

    
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
