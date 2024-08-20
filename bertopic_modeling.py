import argparse
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI as BertOpenAI
import datetime
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
from app.nlp.nlp_metadata import MIN_WORDS_FOR_BERT, MAX_WORDS_FOR_BERT, BERTOPIC_DATA_DIR, BERTOPIC_MODELS_DIR
from app.show_metadata import ShowKey
import app.utils as utils


openai_client = openai.OpenAI(api_key=settings.openai_api_key)
# openai.api_key = settings.openai_api_key

LOG_FILE = 'bertopic_config_tests/bertopic_config_test'
REPORT_FILE = 'bertopic_config_tests/bertopic_config_report'


CONFIG_OPTIONS = {
    # 'narrative_only': [True, False],
    'sentence_transformer_lm': 'all-MiniLM-L12-v2',
    'vec_ngram_low': 1,
    'vec_ngram_high': 2,
    'bertopic_top_n_words': 5,
    'umap_n_neighbors': 5,
    'umap_n_components': 3,
    'umap_min_dist': [0, 0.01, 0.02],
    # 'invalid_umap_metric': ['euclidian', 'seuclidian', 'haversine', 'mahalanobis'],
    # 'unusable_umap_metric': ['wminkowski'],  # wminkowski: best correlation to vector-based topics, but doesn't translate to 3D clusters
    # 'umap_metric': ['braycurtis', 'minkowski', 'canberra', 'manhattan', 'cosine', 'correlation'],
    'umap_metric': ['braycurtis', 'minkowski', 'canberra', 'manhattan', 'cosine', 'correlation'],
    'umap_random_state': [4, 53, 87],
    'hdbscan_min_cluster_size': [25, 50],
    'hdbscan_min_samples': 10,
    # 'mmr_diversity': [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    'mmr_diversity': 0.05,
    # 'representation_model_type': ['Custom', 'MaximalMarginalRelevance', 'KeyBERTInspired', 'BertOpenAI'],
    'topic_count_threshold': 6,
    'topic_ratio_threshold': 0.5,
    'corr_threshold': 0.4,
    'narrative_freq_threshold': 7,
    'match_threshold': 1,
}


def main():
    ts_log = str(datetime.datetime.now())[:19]
    ts_filename = ts_log.replace(' ', '_').replace('-', '').replace(':', '')
    print(f'begin bertopic_modeling at {ts_log}')
    # parse script params
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--umap_metric", "-m", help="UMAP metric", required=False)
    parser.add_argument("--umap_random_state", "-r", help="UMAP random state", required=False)
    parser.add_argument("--umap_min_dist", "-d", help="UMAP min dist", required=False)
    parser.add_argument("--hdbscan_min_cluster_size", "-c", help="HDBSCAN min cluster size", required=False)
    args = parser.parse_args()
    # assign script params to vars
    show_key = args.show_key
    override_config = {}
    if args.umap_metric:
        override_config['umap_metric'] = args.umap_metric.split(',')
    if args.umap_random_state:
        override_config['umap_random_state'] = [int(v) for v in args.umap_random_state.split(',')]
    if args.umap_min_dist:
        override_config['umap_min_dist'] = [float(v) for v in args.umap_min_dist.split(',')]
    if args.hdbscan_min_cluster_size:
        override_config['hdbscan_min_cluster_size'] = [int(v) for v in args.hdbscan_min_cluster_size.split(',')]

    # print(f'attempt huggingface login')
    # login()
    # print(f'huggingface login success')
        
    print(f"override_config={override_config}")
    configs, config_params_to_values = generate_configs(override_config)

    bert_text_inputs, bert_text_sources = generate_bert_text_inputs(ShowKey(show_key), narrative_only=False)

    simple_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
    episodes = simple_episodes_response['episodes']
    episodes_df = pd.DataFrame(episodes)
    sources_df = pd.DataFrame(bert_text_sources, columns=['episode_key', 'speaker_group', 'wc'])
    sources_df = pd.merge(sources_df, episodes_df, on='episode_key')

    log_file = open(f'{LOG_FILE}_{ts_filename}.csv', 'a')  # append mode
    log_file.write('=======================================================================================\n')
    log_file.write(f'Begin new job for {len(configs)} config variations at {ts_log}\n')
    log_file.write(f'config_params_to_values: {config_params_to_values}\n')
    log_file.close()

    report_dicts = []
    i = 1
    for config in configs:
        print(f"processing config {i} of {len(configs)}")
        report_dict = generate_bertopic_models(bert_text_inputs, config, sources_df, ts_filename)
        i += 1
        if report_dict:
            report_dicts.append(report_dict)

    # log summarized output to dataframe csv
    if report_dicts:
        report_df = pd.DataFrame(report_dicts)
        report_file_name = f'{REPORT_FILE}_{ts_filename}.csv'
        report_df.to_csv(report_file_name, sep='\t')


def generate_configs(config_overrides: dict = None) -> tuple[list, dict]:
    # incorporate overrides
    config_options = CONFIG_OPTIONS
    if config_overrides:
        for override_param, override_values in config_overrides.items():
            config_options[override_param] = override_values
    # generate 'flattened' single-value config variants out of multi-value config
    config_params = []
    config_params_to_values = {}
    config_param_value_counts = []
    for config_param, config_values in config_options.items():
        if isinstance(config_values, list):
            config_params.append(config_param)
            config_params_to_values[config_param] = config_values
            config_param_value_counts.append(len(config_values))
    # print(f'config_params={config_params}')
    # print(f'config_params_to_values={config_params_to_values}')
    # print(f'config_param_value_counts={config_param_value_counts}')

    mx = []
    for vc in config_param_value_counts:
        mx.append([i for i in range(vc)])
    config_combos = list(product(*mx))
    # print(f'config_combos={config_combos}')

    configs = []
    for config_combo in config_combos:
        config = generate_config_instance(config_params, config_combo)
        configs.append(config)

    return configs, config_params_to_values


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


def generate_bertopic_models(bert_text_inputs: list, config: dict, sources_df: pd.DataFrame, ts_filename: str) -> dict|None:
    logs = []
    topic_count_threshold = config['topic_count_threshold']
    topic_ratio_threshold = config['topic_ratio_threshold']
    corr_threshold = config['corr_threshold']
    narrative_freq_threshold = config['narrative_freq_threshold']
    match_threshold = config['match_threshold']

    vec_ngram_low = config['vec_ngram_low']
    vec_ngram_high = config['vec_ngram_high']
    # representation_model_type = config['representation_model_type']
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

    #### INITIATLIZE MODELS ####
    # configurable component models
    vectorizer_model = CountVectorizer(ngram_range=(vec_ngram_low, vec_ngram_high), stop_words='english')
    embedding_model = SentenceTransformer(sentence_transformer_lm)
    umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, min_dist=umap_min_dist, metric=umap_metric, random_state=umap_random_state)
    hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples, gen_min_span_tree=True, prediction_data=True)

    # bertopic models for 3 representation model types: MaximalMarginalRelevance, KeyBERTInspired, BertOpenAI
    mmr_representation_model = MaximalMarginalRelevance(diversity=mmr_diversity)
    mmr_bertopic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
                                  representation_model=mmr_representation_model, top_n_words=bertopic_top_n_words, language='english',
                                  calculate_probabilities=True, verbose=True)
    kb_representation_model = KeyBERTInspired()
    kb_bertopic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
                                representation_model=kb_representation_model, top_n_words=bertopic_top_n_words, language='english', 
                                calculate_probabilities=True, verbose=True)
    openai_representation_model = BertOpenAI(openai_client, delay_in_seconds=5, model='gpt-3.5-turbo', chat=True)
    openai_bertopic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
                                     representation_model=openai_representation_model, top_n_words=bertopic_top_n_words, language='english',
                                     calculate_probabilities=True, verbose=True)
    # elif representation_model_type == 'Custom':
    #     bertopic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model,
    #                                 top_n_words=bertopic_top_n_words, language='english', calculate_probabilities=True, verbose=True)
    # else:
    #     bertopic_model = BERTopic(vectorizer_model=vectorizer_model, language='english', calculate_probabilities=True, verbose=True)

    #### FIT MODELS TO TEXT INPUT DATA ####
    try:
        mmr_bertopic_model.fit_transform(bert_text_inputs)
    except Exception as e:
        print(f'Failure to fit_transform against mmr_bertopic_model with config={config}', e)
        return None
    try:
        kb_bertopic_model.fit_transform(bert_text_inputs)
    except Exception as e:
        print(f'Failure to fit_transform against kb_bertopic_model with config={config}', e)
        return None
    try:
        openai_bertopic_model.fit_transform(bert_text_inputs)
    except Exception as e:
        print(f'Failure to fit_transform against openai_bertopic_model with config={config}', e)
        return None

    mmr_bertopic_docs_df = mmr_bertopic_model.get_document_info(bert_text_inputs)
    kb_bertopic_docs_df = kb_bertopic_model.get_document_info(bert_text_inputs)
    openai_bertopic_docs_df = openai_bertopic_model.get_document_info(bert_text_inputs)

    # the 3 {representation_type}_bertopic_docs_dfs differ only on 3 columns: Name, Representation, and Top_n_words
    # consolidate the 3 of them now to streamline remaining steps (using mmr, but could have used kb or openai)
    bertopic_docs_df = mmr_bertopic_docs_df

    # extract cluster keywords from 'Representation' output of MaximalMarginalRelevance and KeyBERTInspired representations
    # same words may appear in both, so de-dupe and preserve aggregate ranking using merge_sorted_lists util func
    bertopic_docs_df['Representation_mmr'] = mmr_bertopic_docs_df['Representation']
    bertopic_docs_df['Representation_kb'] = kb_bertopic_docs_df['Representation']
    bertopic_docs_df['cluster_keywords'] = bertopic_docs_df.apply(lambda x: utils.merge_sorted_lists(x['Representation_mmr'], x['Representation_kb']), axis=1) 

    # extract cluster title from 'Top_n_words' output of BertOpenAI representation 
    bertopic_docs_df['cluster_title'] = openai_bertopic_docs_df['Top_n_words']

    # TODO Any reason to preserve these? 
    # bertopic_docs_df['Name_mmr'] = mmr_bertopic_docs_df['Name']
    # bertopic_docs_df['Name_kb'] = kb_bertopic_docs_df['Name']
    # bertopic_docs_df['Name_openai'] = openai_bertopic_docs_df['Name']
    # # bertopic_docs_df['Representation_mmr'] = mmr_bertopic_docs_df['Representation']
    # # bertopic_docs_df['Representation_kb'] = kb_bertopic_docs_df['Representation']
    # bertopic_docs_df['Representation_openai'] = openai_bertopic_docs_df['Representation']
    # bertopic_docs_df['Top_n_words_mmr'] = mmr_bertopic_docs_df['Top_n_words']
    # bertopic_docs_df['Top_n_words_kb'] = kb_bertopic_docs_df['Top_n_words']
    # bertopic_docs_df['Top_n_words_openai'] = openai_bertopic_docs_df['Top_n_words']

    bertopic_docs_df.drop('Name', axis=1, inplace=True)
    bertopic_docs_df.drop('Representation', axis=1, inplace=True)
    bertopic_docs_df.drop('Top_n_words', axis=1, inplace=True)

    # the 3 {representation_type}_bertopic_topics_dfs have identical counts and mappings, despite having very different contents
    # using mmr (but could have used kb or openai) for initial analysis/results filtering
    bertopic_topics_df = mmr_bertopic_model.get_topic_info()
    # kb_bertopic_topics_df = kb_bertopic_model.get_topic_info()
    # openai_bertopic_topics_df = openai_bertopic_model.get_topic_info()

    #### RESULTS ANALYSIS PT 1 ####
    # filter out data sets that don't meet minimum threshold criteria
    topic_count = len(bertopic_topics_df)
    doc_count_with_topic = bertopic_topics_df[bertopic_topics_df['Topic'] >= 0]['Count'].sum()
    topic_coverage_ratio = doc_count_with_topic / bertopic_topics_df['Count'].sum()
    if topic_count < topic_count_threshold or topic_coverage_ratio < topic_ratio_threshold:
        print(f'Topic results fail to pass filter: either (a) topic_count={topic_count} falls below topic_count_threshold={topic_count_threshold}, ')
        print(f'or (b) topic_coverage_ratio={topic_coverage_ratio} falls below topic_ratio_threshold={topic_ratio_threshold}')
        return None
    
    logs.append(f'topic_count={topic_count}, topic_coverage_ratio={topic_coverage_ratio}')
    report_dict = config.copy()
    report_dict['topic_count'] = topic_count
    report_dict['topic_coverage_ratio'] = topic_coverage_ratio

    #### GENERATE 3D GRAPH ####
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

    # if an episode-narrative maps to multiple clusters, select the one with greater probability * word count
    # the MAX_WORDS_FOR_BERT constraint forced us to slice up narratives into fragments, resulting in some episode-narratives mapping to multiple clusters
    # it was useful to have all text represented for cluster creation, but for persistence each episode-narrative must map to one and only one cluster
    bertopic_docs_df['prob_x_wc'] = bertopic_docs_df['Probability'] * bertopic_docs_df['wc']
    # TODO: set point_size to 'prob_x_wc' or 'Probability'?
    bertopic_docs_df.sort_values(['episode_key', 'speaker_group', 'prob_x_wc'], ascending=[True, True, False])
    # print(f'len(bertopic_docs_df) before de-duping={len(bertopic_docs_df)}')
    last_ekey = None
    last_spkrgrp = None
    for index, row in bertopic_docs_df.iterrows():
        if row['episode_key'] == last_ekey and row['speaker_group'] == last_spkrgrp:
            bertopic_docs_df.drop(index, inplace=True)
        else:
            last_ekey = row['episode_key']
            last_spkrgrp = row['speaker_group']
    # print(f'len(bertopic_docs_df) after de-duping={len(bertopic_docs_df)}')

    # one-hot encoding of topics and speakers
    # Topic: back up as cluster_id, then run get_dummies against column with single topic value 
    bertopic_prefix = 'Topic_'
    bertopic_docs_df['cluster_id'] = bertopic_docs_df['Topic'].apply(lambda x: str(x))
    bertopic_docs_df = pd.get_dummies(bertopic_docs_df, columns=['Topic'], dtype=int, drop_first=False)
    
    # vector topics and speaker_group: convert to list, then one-hot encode values from the resulting 'list' column 
    topics_focused_prefix = 'foctopic_'
    topics_focused_tfidf_prefix = 'foctopictfidf_'
    topics_universal_prefix = 'univtopic_'
    topics_universal_tfidf_prefix = 'univtopictfidf_'
    speaker_prefix = 'spkr_'
    # topics_focused
    bertopic_docs_df['topics_focused_list'] = bertopic_docs_df['topics_focused'].apply(lambda x: [t['topic_key'] for t in x[:3]])
    bertopic_docs_df = one_hot_multival_col(bertopic_docs_df, topics_focused_prefix, 'topics_focused_list')
    # topics_focused_tfidf
    bertopic_docs_df['topics_focused_tfidf_list'] = bertopic_docs_df['topics_focused_tfidf'].apply(lambda x: [t['topic_key'] for t in x[:3]])
    bertopic_docs_df = one_hot_multival_col(bertopic_docs_df, topics_focused_tfidf_prefix, 'topics_focused_tfidf_list')
    # topics_universal
    bertopic_docs_df['topics_universal_list'] = bertopic_docs_df['topics_universal'].apply(lambda x: [t['topic_key'] for t in x[:3]])
    bertopic_docs_df = one_hot_multival_col(bertopic_docs_df, topics_universal_prefix, 'topics_universal_list')
    # topics_universal_tfidf
    bertopic_docs_df['topics_universal_tfidf_list'] = bertopic_docs_df['topics_universal_tfidf'].apply(lambda x: [t['topic_key'] for t in x[:3]])
    bertopic_docs_df = one_hot_multival_col(bertopic_docs_df, topics_universal_tfidf_prefix, 'topics_universal_tfidf_list')
    # speaker_group 
    bertopic_docs_df['speaker_group_list'] = bertopic_docs_df['speaker_group'].apply(lambda x: x.split('_'))
    bertopic_docs_df = one_hot_multival_col(bertopic_docs_df, speaker_prefix, 'speaker_group_list')

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

    #### RESULTS ANALYSIS PT 2 ####
    # generate corr dfs
    bertopic_cols = [c for c in bertopic_docs_df.columns if bertopic_prefix in c]
    foctopic_cols = [c for c in bertopic_docs_df.columns if topics_focused_prefix in c]
    foctopictfidf_cols = [c for c in bertopic_docs_df.columns if topics_focused_tfidf_prefix in c]
    univtopic_cols = [c for c in bertopic_docs_df.columns if topics_universal_prefix in c]
    univtopictfidf_cols = [c for c in bertopic_docs_df.columns if topics_universal_tfidf_prefix in c]
    speaker_cols = [c for c in bertopic_docs_df.columns if speaker_prefix in c]

    # bertopic_cols corrs to vector topic and speaker cols
    # save_models = False
    corr_values_foctopic_x_bertopic = extract_corr_values(bertopic_docs_df, foctopic_cols, bertopic_cols, corr_threshold, narrative_freq_threshold)
    if len(corr_values_foctopic_x_bertopic) >= match_threshold:
        logs.append(f'corr_values_foctopic_x_bertopic={corr_values_foctopic_x_bertopic}')
        report_dict['foctopic_x_bertopic'] = len(corr_values_foctopic_x_bertopic)
        # save_models = True
    else:
        report_dict['foctopic_x_bertopic'] = 0
    corr_values_foctopictfidf_x_bertopic = extract_corr_values(bertopic_docs_df, foctopictfidf_cols, bertopic_cols, corr_threshold, narrative_freq_threshold)
    if len(corr_values_foctopictfidf_x_bertopic) >= match_threshold:
        logs.append(f'corr_values_foctopictfidf_x_bertopic={corr_values_foctopictfidf_x_bertopic}')
        report_dict['foctopictfidf_x_bertopic'] = len(corr_values_foctopictfidf_x_bertopic)
        # save_models = True
    else:
        report_dict['foctopictfidf_x_bertopic'] = 0
    corr_values_univtopic_x_bertopic = extract_corr_values(bertopic_docs_df, univtopic_cols, bertopic_cols, corr_threshold, narrative_freq_threshold)
    if len(corr_values_univtopic_x_bertopic) >= match_threshold:
        logs.append(f'corr_values_univtopic_x_bertopic={corr_values_univtopic_x_bertopic}')
        report_dict['univtopic_x_bertopic'] = len(corr_values_univtopic_x_bertopic)
        # save_models = True
    else:
        report_dict['univtopic_x_bertopic'] = 0
    corr_values_univtopictfidf_x_bertopic = extract_corr_values(bertopic_docs_df, univtopictfidf_cols, bertopic_cols, corr_threshold, narrative_freq_threshold)
    if len(corr_values_univtopictfidf_x_bertopic) >= match_threshold:
        logs.append(f'corr_values_univtopictfidf_x_bertopic={corr_values_univtopictfidf_x_bertopic}')
        report_dict['univtopictfidf_x_bertopic'] = len(corr_values_univtopictfidf_x_bertopic)
        # save_models = True
    else:
        report_dict['univtopictfidf_x_bertopic'] = 0
    corr_values_speaker_x_bertopic = extract_corr_values(bertopic_docs_df, speaker_cols, bertopic_cols, corr_threshold, narrative_freq_threshold)
    if len(corr_values_speaker_x_bertopic) >= match_threshold:
        logs.append(f'corr_values_speaker_x_bertopic={corr_values_speaker_x_bertopic}')
        report_dict['speaker_x_bertopic'] = len(corr_values_speaker_x_bertopic)
        # save_models = True
    else:
        report_dict['speaker_x_bertopic'] = 0

    #### LOGGING ####
    print('-----------------------------------------------------------------------------------------')
    print(f'config={config}')
    print(f'logs={logs}')
    f = open(LOG_FILE, 'a')  # append mode
    f.write('-----------------------------------------------------------------------------------------\n')
    f.write(f'config: {config}\n')
    for log in logs:
        f.write(f'*** {log}\n')
    f.close()

    #### SAVE MODEL DATA ####
    # if save_models:
    model_id = f'{umap_metric}_{umap_random_state}_{umap_min_dist}_{hdbscan_min_cluster_size}'
    # create model dirs
    os.mkdir(f"{BERTOPIC_MODELS_DIR}/{model_id}")
    os.mkdir(f"{BERTOPIC_MODELS_DIR}/{model_id}/mmr")
    os.mkdir(f"{BERTOPIC_MODELS_DIR}/{model_id}/kb")
    os.mkdir(f"{BERTOPIC_MODELS_DIR}/{model_id}/openai")
    # save models
    se_model = f"sentence-transformers/{sentence_transformer_lm}"
    mmr_bertopic_model.save(f"{BERTOPIC_MODELS_DIR}/{model_id}/mmr", serialization="safetensors", save_ctfidf=True, save_embedding_model=se_model)
    kb_bertopic_model.save(f"{BERTOPIC_MODELS_DIR}/{model_id}/kb", serialization="safetensors", save_ctfidf=True, save_embedding_model=se_model)
    openai_bertopic_model.save(f"{BERTOPIC_MODELS_DIR}/{model_id}/openai", serialization="safetensors", save_ctfidf=True, save_embedding_model=se_model)
    # save bertopic_docs_df
    bertopic_docs_file_name = f'{BERTOPIC_DATA_DIR}/{model_id}.csv'
    print(f'writing bertopic_docs_df to file path={bertopic_docs_file_name}')
    bertopic_docs_df.to_csv(bertopic_docs_file_name, sep='\t')

    return report_dict

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


def one_hot_multival_col(bertopic_docs_df: pd.DataFrame, prefix: str, multi_val_list_col: str) -> pd.DataFrame:
    distinct_vals = sorted(set(chain.from_iterable(bertopic_docs_df[multi_val_list_col])))
    distinct_val_series = pd.Series(distinct_vals, index=[f'{prefix}{s}' for s in distinct_vals])
    bertopic_docs_df = bertopic_docs_df.join(bertopic_docs_df[multi_val_list_col].apply(lambda x: distinct_val_series.isin(x)).astype(int))
    return bertopic_docs_df


def extract_corr_values(bertopic_docs_df: pd.DataFrame, corr_cols: list, bertopic_cols: list, threshold: float, narrative_freq_threshold: int) -> list:
    # trim df down to just the columns to be correlated, then further trim out corr_cols with fewer than `narrative_freq_threshold` occurrences
    # bertopic_docs_df_trimmed = bertopic_docs_df[corr_cols + bertopic_cols]
    print(f'bertopic_docs_df[corr_cols].sum()={bertopic_docs_df[corr_cols].sum()}')
    freq_corr_cols = bertopic_docs_df[corr_cols].sum() > narrative_freq_threshold
    freq_corr_cols_list = list(freq_corr_cols[freq_corr_cols].keys())
    bertopic_docs_df_trimmed = bertopic_docs_df[freq_corr_cols_list + bertopic_cols]
    print(f'freq_corr_cols_list={freq_corr_cols_list} bertopic_docs_df_trimmed.columns={bertopic_docs_df_trimmed.columns}')
    # generate corr df, then remove duplicate cells on different axes from df
    corr_df = bertopic_docs_df_trimmed.corr()
    for col_name in corr_df.columns:
        if col_name in corr_cols:
            corr_df.drop(col_name, axis=1, inplace=True)
    for i, _ in corr_df.iterrows():
        if i in bertopic_cols:
            corr_df.drop(i, inplace=True)

    # return corr columns and values as list
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


# def log_and_save_topics(model: BERTopic, model_name: str, bert_text_inputs: list, sources_df: pd.DataFrame):
#     print('----------------------------------------------------------------------------')
#     print(f'BEGIN REPORT for model_name={model_name}')
#     topic_freq = model.get_topic_info()
#     print(f'topic_freq={topic_freq}')
#     topic_freq.to_csv(f'topic_data/topic_freq_{model_name}.csv')

#     for i in range(topic_freq['Topic'].max()):
#         print(f'model.get_topic({i})={model.get_topic(i)}')

#     doc_info = model.get_document_info(bert_text_inputs)
#     print(f'doc_info={doc_info}')
#     doc_info = pd.concat([doc_info, sources_df], axis=1)
#     doc_info['episode_key'] = doc_info['source'].apply(lambda x: x.split('_')[0])
#     doc_info['prob_x_wc'] = doc_info['Probability'] * doc_info['wc']
#     doc_info.to_csv(f'topic_data/doc_info_{model_name}.csv')

#     # drop topic -1
#     doc_info = doc_info[doc_info['Topic'] > -1]
#     # assemble tuple lists mapping episodes to topics
#     topic_episode_rows = []
#     for i in range(doc_info['Topic'].max()):
#         temp_df = doc_info[doc_info['Topic'] == i]
#         e_keys = list(temp_df['episode_key'].unique())
#         for e_key in e_keys:
#             e_key_df = temp_df[temp_df['episode_key'] == e_key]
#             topic_episode_rows.append(dict(topic=i, episode_key=e_key, mapping_cnt=len(e_key_df), high_prob=e_key_df['Probability'].max(), sum_weighted_probs=e_key_df['prob_x_wc'].sum())) 

#     topic_episode_df = pd.DataFrame(topic_episode_rows)
#     topic_episode_df = topic_episode_df.sort_values(['topic', 'sum_weighted_probs'], ascending=[True, False])
#     topic_episode_df.to_csv(f'topic_data/topic_episode_{model_name}.csv')

#     # model.push_to_hf_hub(repo_id=f'andyshirey/test_bert_{model_name}', save_ctfidf=True, save_embedding_model=embed_model, serialization='pytorch')
#     model.save(f'topic_models/model_{model_name}', serialization='safetensors')


# def dumped():
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
