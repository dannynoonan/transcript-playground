import argparse
import datetime
from itertools import product
import os
import pandas as pd
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from app.app_metadata import BERTOPIC_LOGS_DIR
import app.routers.es_read_router as esr
import app.nlp.bertopic_model_builder as bmb
from app.show_metadata import ShowKey


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

    # TODO haven't solved for setting this correctly, requires altering exit_if_unauthorized to run 
    user_dependency = None

    bert_text_inputs, bert_text_sources = bmb.generate_bert_text_inputs(ShowKey(show_key), narrative_only=False)

    simple_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key), user_dependency)
    episodes = simple_episodes_response['episodes']
    episodes_df = pd.DataFrame(episodes)
    sources_df = pd.DataFrame(bert_text_sources, columns=['episode_key', 'speaker_group', 'wc'])
    sources_df = pd.merge(sources_df, episodes_df, on='episode_key')

    log_file = open(f'{BERTOPIC_LOGS_DIR}/log_{show_key}_{ts_filename}', 'a')  # append mode
    log_file.write('=======================================================================================\n')
    log_file.write(f'Begin new job for {len(configs)} config variations at {ts_log}\n')
    log_file.write(f'config_params_to_values: {config_params_to_values}\n')
    log_file.close()

    report_dicts = []
    i = 1
    for config in configs:
        print(f"processing config {i} of {len(configs)}")
        report_dict = bmb.generate_bertopic_models(show_key, bert_text_inputs, config, sources_df)
        i += 1
        if report_dict:
            report_dicts.append(report_dict)

    # log summarized output to dataframe csv
    if report_dicts:
        report_df = pd.DataFrame(report_dicts)
        report_file_name = f'{BERTOPIC_LOGS_DIR}/report_{show_key}_{ts_filename}.csv'
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


if __name__ == '__main__':
    main()
