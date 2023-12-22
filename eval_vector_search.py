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

import main as m
from show_metadata import ShowKey
from soup_brewer import get_episode_description_soup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--model_vendor", "-m", help="Model vendor", required=True)
    parser.add_argument("--model_version", "-v", help="Model version", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    model_vendor = args.model_vendor
    model_version = args.model_version
    print(f'begin eval_vector_search script for show_key={show_key} vendor={model_vendor} version={model_version}')

    file_path = f'./analytics/model_rankings_{show_key}_TEST.csv'
    if os.path.isfile(file_path):
        episodes_df = pd.read_csv(file_path, '\t')
        print(f'loading episodes dataframe from file found at file_path={file_path}')

    else:
        print(f'no file found at file_path={file_path}, initializing new episodes dataframe')
        episodes_df = init_episode_df(show_key)
        scrape_episode_descriptions(episodes_df, 'memory_alpha')
        scrape_episode_descriptions(episodes_df, 'johanw')

    # generate_vector_search_rankings(episodes_df, show_key, model_vendor, model_version)

    print(f'episodes_df={episodes_df}')

    episodes_df.to_csv(file_path, sep='\t')


def init_episode_df(show_key: str) -> pd.DataFrame:
    episodes_by_season_resp = m.list_episodes_by_season(ShowKey(show_key))
    episodes_by_season = episodes_by_season_resp['episodes_by_season']
    
    episodes_list = []
    for season, episodes in episodes_by_season.items():
        for e in episodes:
            e['title_upper'] = e['title'].upper()
            e['title_camel'] = e['title'].title()
            episodes_list.append(e)
    episodes_df = pd.DataFrame(episodes_list)
    
    return episodes_df


def scrape_episode_descriptions(episodes_df: pd.DataFrame, description_source: str) -> None:
    print(f'begin scrape_episode_descriptions for {len(episodes_df)} episodes from description_source={description_source}')
    descr_col = f'description_{description_source}'
    episodes_df[descr_col] = ''
    ds_url = DESCRIPTION_SOURCES[description_source]

    if description_source == 'johanw':
        episode_desc_soup = get_episode_description_soup(ds_url)
        episodes = [h3_tag for h3_tag in episode_desc_soup.find_all('h3')]
        for e in episodes:
            desc_p = e.find_next('p')
            desc_text = desc_p.get_text()
            while (desc_p.next_element.next_element.name == 'p'):
                desc_p = desc_p.next_element.next_element
                desc_text += desc_p.get_text()

            title = e.get_text()
            if title in TITLE_VARIATIONS:
                title = TITLE_VARIATIONS[title]

            if (episodes_df.title_upper == title).any():
                episodes_df.loc[episodes_df['title_upper'] == title, descr_col] = desc_text.replace('\n', ' ')
            else:
                print(f'No title match for e.get_text()={e.get_text()} or title={title}')

    elif description_source == 'memory_alpha':
        for index, row in episodes_df.iterrows():
            title_undscr = row['title'].replace(' ', '_')
            title_camel_undscr = row['title_camel'].replace(' ', '_')
            request_url = f'{ds_url}{title_undscr}_(episode)'
            alt_request_url = f'{ds_url}{title_camel_undscr}_(episode)'
            try:
                episode_desc_soup = get_episode_description_soup(request_url)
            except Exception:
                print(f'failed to find description for episode_key={row["episode_key"]} at request_url={request_url}: url was invalid. skipping.')
                continue
            h2_tags = episode_desc_soup.find_all('h2')
            if not h2_tags or len(h2_tags) < 3:
                try:
                    episode_desc_soup = get_episode_description_soup(alt_request_url)
                except Exception:
                    print(f'failed to find description for episode_key={row["episode_key"]} at alt_request_url={alt_request_url}: url was invalid. skipping.')
                    continue
                h2_tags = episode_desc_soup.find_all('h2')
                if not h2_tags or len(h2_tags) < 3:
                    print(f'failed to find description for episode_key={row["episode_key"]} at both request_url={request_url} and alt_request_url={alt_request_url}: skipping.')
                    continue
            desc_p = h2_tags[2].find_next_sibling('p')
            if not desc_p:
                print(f'failed to find description for episode_key={row["episode_key"]} at request_url={request_url}: problem extracting p tag after h2 tag. skipping.')
                continue
            desc_text = desc_p.get_text()
            while (desc_p.next_element.next_element.name == 'p'):
                desc_p = desc_p.next_element.next_element
                desc_text += desc_p.get_text()
            print(f'--> SUCCESS finding episode description for episode_key={row["episode_key"]} at request_url={request_url}:')
            episodes_df.loc[episodes_df['title'] == row['title'], descr_col] = desc_text.replace('\n', ' ')


def generate_vector_search_rankings(episodes_df: pd.DataFrame, show_key: str, model_vendor: str, model_version: str) -> None:
    print(f'begin generate_vector_search_rankings for model_vendor={model_vendor} model_version={model_version}')

    rank_col = f'rank_{model_vendor}_{model_version}'
    score_col = f'score_{model_vendor}_{model_version}'
    matched_tokens_col = f'matched_tokens_{model_vendor}_{model_version}'
    matched_tokens_count_col = f'matched_tokens_count_{model_vendor}_{model_version}'
    unmatched_tokens_col = f'unmatched_tokens_{model_vendor}_{model_version}'
    unmatched_tokens_count_col = f'unmatched_tokens_count_{model_vendor}_{model_version}'
    episodes_df[rank_col] = ''
    episodes_df[matched_tokens_col] = ''
    episodes_df[matched_tokens_count_col] = ''
    episodes_df[unmatched_tokens_col] = ''
    episodes_df[unmatched_tokens_count_col] = ''

    success_count = 0
    for index, row in episodes_df.iterrows():
        if not row['description']:
            print(f"description is empty for episode_key={row['episode_key']}, skipping")
            continue
        episode_key = row['episode_key']
        vector_search_response = m.vector_search(ShowKey(show_key), row['description'], model_vendor=model_vendor, model_version=model_version)
        if 'error' in vector_search_response:
            print(f"Failed to generate_vector_search_rankings for episode_key={episode_key}: {vector_search_response['error']}")
            continue
        rank = 1
        found = False
        for match in vector_search_response['matches']:
            if match['episode_key'] == str(episode_key):
                episodes_df.loc[episodes_df['episode_key'] == episode_key, rank_col] = rank
                episodes_df.loc[episodes_df['episode_key'] == episode_key, score_col] = match['agg_score']
                found = True
                break
            rank += 1
        if not found:
            print(f"no match found for episode_key={row['episode_key']}, setting rank to max and score to 0")
        episodes_df.loc[episodes_df['episode_key'] == episode_key, matched_tokens_col] = ', '.join(vector_search_response['tokens_processed'])
        episodes_df.loc[episodes_df['episode_key'] == episode_key, matched_tokens_count_col] = len(vector_search_response['tokens_processed'])
        episodes_df.loc[episodes_df['episode_key'] == episode_key, unmatched_tokens_col] = ', '.join(vector_search_response['tokens_failed'])
        episodes_df.loc[episodes_df['episode_key'] == episode_key, unmatched_tokens_count_col] = len(vector_search_response['tokens_failed'])
        success_count += 1
    
    print(f'completed generate_vector_search_rankings for {success_count} episodes')


TITLE_VARIATIONS = {'FIRST BORN': 'FIRSTBORN', 'DESCENT': 'DESCENT, PART I', 'INNER LIGHT': 'THE INNER LIGHT', 'FISTFUL OF DATAS': 'A FISTFUL OF DATAS',
                    'THE BEST OF BOTH WORLDS': 'THE BEST OF BOTH WORLDS, PART I', 'BEST OF BOTH WORLDS, PART II': 'THE BEST OF BOTH WORLDS, PART II',
                    'DATA`S DAY': 'DATA\'S DAY', 'HALF-LIFE': 'HALF A LIFE', 'REDEMPTION': 'REDEMPTION, PART I', 'REDEMPTION II': 'REDEMPTION, PART II',
                    'UNIFICATION I': 'UNIFICATION, PART I', 'UNIFICATION II': 'UNIFICATION, PART II', 'THE COST OF LIVING': 'COST OF LIVING',
                    'INNER LIGHT': 'THE INNER LIGHT', 'TIME\'S ARROW': 'TIME\'S ARROW, PART I', 'DESCENT': 'DESCENT, PART I', 
                    'All GOOD THINGS, PART I': 'ALL GOOD THINGS...', 'ALL GOOD THINGS, PART II': 'ALL GOOD THINGS...'}
    

DESCRIPTION_SOURCES = {
    'memory_alpha': 'https://memory-alpha.fandom.com/wiki/',
    'johanw': 'https://johanw.home.xs4all.nl/sttng.html'
}


if __name__ == '__main__':
    main()
