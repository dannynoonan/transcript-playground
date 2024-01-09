'''
* Scan https://johanw.home.xs4all.nl/sttng.html, Load Title and Description into dataframe
* Pair episode_key with each episode in df
* Run vector search against all episodes / all models
* Add a column to df for each model and insert the episode rank in the model's vector search results
* Add another 2 columns to df for each model and insert matched tokens and unmatched tokens per episode
* Keep track of which tokens are unmatched most frequently
* Average out rankings, determine if any model is a clear winner 
'''

import argparse
import os
import pandas as pd
import time

from load_description_sources import DESCRIPTION_SOURCES
import main as m
from nlp.nlp_metadata import WORD2VEC_VENDOR_VERSIONS, ACTIVE_VENDOR_VERSIONS
from show_metadata import ShowKey


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--desc_source", "-d", help="Description source", required=True)
    parser.add_argument("--model_vendor", "-m", help="Model vendor", required=True)
    parser.add_argument("--model_version", "-v", help="Model version", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    desc_source = args.desc_source
    model_vendor = args.model_vendor
    model_version = args.model_version
    print(f'begin eval_vector_search script for show_key={show_key} desc_source={desc_source} model_vendor={model_vendor} model_version={model_version}')

    episode_desc_file_path = f'./analytics/desc_sources_{show_key}.csv'
    if not os.path.isfile(episode_desc_file_path):
        print(f'no file found at episode_desc_file_path={episode_desc_file_path}, please run `load_description_sources` first')
        exit()

    episode_desc_df = pd.read_csv(episode_desc_file_path, '\t')
    print(f'loading description source dataframe from file found at episode_desc_file_path={episode_desc_file_path}')
    if desc_source not in episode_desc_df.columns:
        print(f'no column for desc_source={desc_source} found in episode_desc_file_path={episode_desc_file_path}, please run `load_description_sources` for desc_source first')
        exit()

    episode_rank_file_path = f'./analytics/model_rankings_{show_key}_{desc_source}.csv'
    if not os.path.isfile(episode_rank_file_path):
        episode_rank_df = episode_desc_df.copy(deep=False)
        cols_to_remove = dict(DESCRIPTION_SOURCES)
        del cols_to_remove[desc_source]
        episode_rank_df.drop(cols_to_remove.keys(), axis=1, inplace=True)
        # current_ts = time.strftime("%Y%m%d-%H%M%S")
        # os.rename(episode_rank_file_path, f'./analytics/model_rankings_{show_key}_{desc_source}_{current_ts}.csv')
    else:
        episode_rank_df = pd.read_csv(episode_rank_file_path, '\t')

    if model_vendor == 'ALL':
        for vendor_version in ACTIVE_VENDOR_VERSIONS:
            generate_vector_search_rankings(episode_rank_df, desc_source, show_key, vendor_version[0], vendor_version[1])
    elif model_version == 'ALL':
        for vendor_version in ACTIVE_VENDOR_VERSIONS:
            if vendor_version[0] == model_vendor:
                generate_vector_search_rankings(episode_rank_df, desc_source, show_key, vendor_version[0], vendor_version[1])
    else:
        generate_vector_search_rankings(episode_rank_df, desc_source, show_key, model_vendor, model_version)

    for col in episode_rank_df.columns:
        if 'Unnamed' in col:
            episode_rank_df.drop(col, axis=1, inplace=True)
    print(f'episode_rank_df={episode_rank_df}')

    episode_rank_df.to_csv(episode_rank_file_path, sep='\t')


def generate_vector_search_rankings(episode_rank_df: pd.DataFrame, desc_source: str, show_key: str, model_vendor: str, model_version: str) -> None:
    print(f'begin generate_vector_search_rankings for model_vendor={model_vendor} model_version={model_version}')

    rank_col = f'rank_{model_vendor}_{model_version}'
    score_col = f'score_{model_vendor}_{model_version}'
    matched_tokens_count_col = f'matched_tokens_count_{model_vendor}_{model_version}'
    unmatched_tokens_count_col = f'unmatched_tokens_count_{model_vendor}_{model_version}'
    episode_rank_df[rank_col] = ''
    episode_rank_df[matched_tokens_count_col] = ''
    episode_rank_df[unmatched_tokens_count_col] = ''
    if model_vendor in WORD2VEC_VENDOR_VERSIONS:
        matched_tokens_col = f'matched_tokens_{model_vendor}_{model_version}'
        unmatched_tokens_col = f'unmatched_tokens_{model_vendor}_{model_version}'
        episode_rank_df[matched_tokens_col] = ''
        episode_rank_df[unmatched_tokens_col] = ''

    success_count = 0
    for _, row in episode_rank_df.iterrows():
        if not row[desc_source]:
            print(f"description field `{desc_source}` is empty for episode_key={row['episode_key']}, skipping")
            continue
        episode_key = row['episode_key']
        vector_search_response = m.vector_search(ShowKey(show_key), row[desc_source], model_vendor=model_vendor, model_version=model_version)
        if 'error' in vector_search_response:
            print(f"Failed to generate_vector_search_rankings for episode_key={episode_key}: {vector_search_response['error']}")
            continue
        rank = 1
        found = False
        for match in vector_search_response['matches']:
            if match['episode_key'] == str(episode_key):
                episode_rank_df.loc[episode_rank_df['episode_key'] == episode_key, rank_col] = int(rank)
                episode_rank_df.loc[episode_rank_df['episode_key'] == episode_key, score_col] = match['agg_score']
                found = True
                break
            rank += 1
        if not found:
            print(f"no match found for episode_key={episode_key}, leaving rank and score fields empty")
        episode_rank_df.loc[episode_rank_df['episode_key'] == episode_key, matched_tokens_count_col] = vector_search_response['tokens_processed_count']
        episode_rank_df.loc[episode_rank_df['episode_key'] == episode_key, unmatched_tokens_count_col] = vector_search_response['tokens_failed_count']
        if model_vendor in WORD2VEC_VENDOR_VERSIONS:
            episode_rank_df.loc[episode_rank_df['episode_key'] == episode_key, matched_tokens_col] = ', '.join(vector_search_response['tokens_processed'])
            episode_rank_df.loc[episode_rank_df['episode_key'] == episode_key, unmatched_tokens_col] = ', '.join(vector_search_response['tokens_failed'])
        success_count += 1
    
    print(f'completed generate_vector_search_rankings for {success_count} episodes')


if __name__ == '__main__':
    main()
