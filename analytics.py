'''
* Scan https://johanw.home.xs4all.nl/sttng.html, Load Title and Description into dataframe
* Pair episode_key with each episode in df
* Run vector search against all episodes / all models
* Add a column to df for each model and insert the episode rank in the model's vector search results
* Add another 2 columns to df for each model and insert matched tokens and unmatched tokens per episode
* Keep track of which tokens are unmatched most frequently
* Average out rankings, determine if any model is a clear winner 
'''

import pandas as pd

import main as m
from show_metadata import ShowKey
from soup_brewer import get_episode_description_soup


def main():
    episodes_df = init_episode_df()
    scrape_episode_descriptions(episodes_df)
    generate_vector_search_rankings(episodes_df, 'webvectors', 'gigaword29')
    generate_vector_search_rankings(episodes_df, 'webvectors', 'enwiki223')
    generate_vector_search_rankings(episodes_df, 'glove', '6B300d')
    generate_vector_search_rankings(episodes_df, 'glove', 'twitter27B200d')
    generate_vector_search_rankings(episodes_df, 'glove', 'twitter27B100d')
    generate_vector_search_rankings(episodes_df, 'fasttext', 'wikinews300d1M')
    episodes_df.to_csv('./analytics/description_embeddings_rankings.csv', sep='\t')


def init_episode_df() -> pd.DataFrame:
    episodes_by_season_resp = m.list_episodes_by_season(ShowKey('TNG'))
    episodes_by_season = episodes_by_season_resp['episodes_by_season']
    
    episodes_list = []
    for season, episodes in episodes_by_season.items():
        for e in episodes:
            e['title'] = e['title'].upper()
            episodes_list.append(e)
    episodes_df = pd.DataFrame(episodes_list)
    
    return episodes_df


def scrape_episode_descriptions(episodes_df: pd.DataFrame) -> None:
    print(f'begin scrape_episode_descriptions for {len(episodes_df)} episodes')
    episode_desc_soup = get_episode_description_soup('https://johanw.home.xs4all.nl/sttng.html')
    episodes = [h3_tag for h3_tag in episode_desc_soup.find_all('h3')]
    episodes_df['description'] = ''
    for e in episodes:
        desc_p = e.find_next('p')
        desc_text = desc_p.get_text()
        while (desc_p.next_element.next_element.name == 'p'):
            desc_p = desc_p.next_element.next_element
            desc_text += desc_p.get_text()

        title = e.get_text()
        if title in TITLE_VARIATIONS:
            title = TITLE_VARIATIONS[title]

        if (episodes_df.title == title).any():
            episodes_df.loc[episodes_df['title'] == title, 'description'] = desc_text.replace('\n', ' ')
        else:
            print(f'No title match for e.get_text()={e.get_text()} or title={title}')


def generate_vector_search_rankings(episodes_df: pd.DataFrame, model_vendor: str, model_version: str) -> None:
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
        vector_search_response = m.vector_search(ShowKey('TNG'), row['description'], model_vendor=model_vendor, model_version=model_version)
        if 'error' in vector_search_response:
            print(f"Failed to generate_vector_search_rankings for episode_key={episode_key}: {vector_search_response['error']}")
            continue
        rank = 1
        found = False
        for match in vector_search_response['matches']:
            if match['episode_key'] == episode_key:
                print(f"found match for episode_key={row['episode_key']} at rank={rank} with score={match['agg_score']}")
                episodes_df.loc[episodes_df['episode_key'] == episode_key, rank_col] = rank
                episodes_df.loc[episodes_df['episode_key'] == episode_key, score_col] = match['agg_score']
                # row[rank_col] = rank
                # row[score_col] = match['agg_score']
                found = True
                break
            rank += 1
        if not found:
            print(f"no match found for episode_key={row['episode_key']}, setting rank to max and score to 0")
            # episodes_df.loc[episodes_df['episode_key'] == episode_key, rank_col] = 999999
            # episodes_df.loc[episodes_df['episode_key'] == episode_key, score_col] = 0
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
    

if __name__ == '__main__':
    main()
