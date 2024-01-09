import argparse
import os
import pandas as pd

import main as m
import show_metadata as sm
import source_etl.soup_brewer as sb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--desc_source", "-d", help="Description source", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    desc_source = args.desc_source
    print(f'begin load_description_sources script for show_key={show_key} desc_source={desc_source}')

    file_path = f'./analytics/desc_sources_{show_key}.csv'
    if os.path.isfile(file_path):
        episodes_df = pd.read_csv(file_path, '\t')
        print(f'loading episodes dataframe from file found at file_path={file_path}')
    else:
        print(f'no file found at file_path={file_path}, initializing new episodes dataframe')
        episodes_df = init_episode_df(show_key)
        
    scrape_episode_descriptions(episodes_df, desc_source)

    for col in episodes_df.columns:
        if 'Unnamed' in col:
            episodes_df.drop(col, axis=1, inplace=True)
    print(f'episodes_df={episodes_df}')

    episodes_df.to_csv(file_path, sep='\t')


def init_episode_df(show_key: str) -> pd.DataFrame:
    episodes_by_season_resp = m.list_episodes_by_season(sm.ShowKey(show_key))
    episodes_by_season = episodes_by_season_resp['episodes_by_season']
    
    episodes_list = []
    for _, episodes in episodes_by_season.items():
        for e in episodes:
            e['title_upper'] = e['title'].upper()
            title_camel = e['title'].title()
            if ' Ii' in title_camel:
                title_camel.replace(' Ii', ' II')
            e['title_camel'] = title_camel
            episodes_list.append(e)
    episodes_df = pd.DataFrame(episodes_list)
    episodes_df.drop(['show_key', 'scene_count', 'indexed_ts'], axis=1, inplace=True)
    
    return episodes_df


def scrape_episode_descriptions(episodes_df: pd.DataFrame, description_source: str) -> None:
    print(f'begin scrape_episode_descriptions for {len(episodes_df)} episodes from description_source={description_source}')
    descr_col = description_source
    episodes_df[descr_col] = ''
    ds_url = DESCRIPTION_SOURCES[description_source]

    if description_source == 'johanw':
        episode_desc_soup = sb.get_episode_description_soup(ds_url)
        episodes = [h3_tag for h3_tag in episode_desc_soup.find_all('h3')]
        for e in episodes:
            desc_p = e.find_next('p')
            desc_text = desc_p.get_text()
            while (desc_p.next_element.next_element.name == 'p'):
                desc_p = desc_p.next_element.next_element
                desc_text += desc_p.get_text()

            title = e.get_text()
            if title in JOHANW_TITLE_VARIATIONS:
                title = JOHANW_TITLE_VARIATIONS[title]

            if (episodes_df.title_upper == title).any():
                episodes_df.loc[episodes_df['title_upper'] == title, descr_col] = desc_text.replace('\n', ' ')
            else:
                print(f'No title match for e.get_text()={e.get_text()} or title={title}')

    elif description_source == 'memory_alpha':
        successful = 0
        failed = []
        for _, row in episodes_df.iterrows():
            request_urls = []
            if row['title'] in MEMORY_ALPHA_TITLE_VARIATIONS:
                title_undscr = MEMORY_ALPHA_TITLE_VARIATIONS[row['title']].replace(' ', '_')
                request_urls.append(f'{ds_url}{title_undscr}_(episode)')
            else:
                title_undscr = row['title'].replace(' ', '_')
                title_camel_undscr = row['title_camel'].replace(' ', '_')
                request_urls.append(f'{ds_url}{title_undscr}_(episode)')
                request_urls.append(f'{ds_url}{title_camel_undscr}_(episode)')
                if ', Part' in row['title']:
                    title_undscr_trim = title_undscr.replace(',_Part', '')
                    request_urls.append(f'{ds_url}{title_undscr_trim}_(episode)')
                    if ' I' in row['title'] and ' II' not in row['title']:
                        title_undscr_trimmer = title_undscr_trim.replace('_I', '')
                        request_urls.append(f'{ds_url}{title_undscr_trimmer}_(episode)')
                    title_camel_undscr_trim = title_camel_undscr.replace(',_Part', '')
                    request_urls.append(f'{ds_url}{title_camel_undscr_trim}_(episode)')
                    if ' I' in row['title'] and ' II' not in row['title']:
                        title_camel_undscr_trimmer = title_camel_undscr_trim.replace('_I', '')
                        request_urls.append(f'{ds_url}{title_camel_undscr_trimmer}_(episode)')
            found = False
            i = 0
            while not found and i < len(request_urls):
                request_url = request_urls[i]
                try:
                    episode_desc_soup = sb.get_episode_description_soup(request_url)
                except Exception:
                    print(f'unable to find description for episode_key={row["episode_key"]} at request_url={request_url}: url was invalid.')
                h2_tags = episode_desc_soup.find_all('h2')
                if h2_tags and len(h2_tags) > 2:
                    found = True
                else:
                    print(f'unable to find description for episode_key={row["episode_key"]} at request_url={request_url}: url was valid but did not load episode.')
                i += 1
            if not found or not h2_tags[2].find_next_sibling('p'):
                print(f'----> FAILED to find description for episode_key={row["episode_key"]} title={row["title"]} at any request_url. skipping.')
                failed.append(f'{row["episode_key"]}:{row["title"]}')
                continue
            desc_p = h2_tags[2].find_next_sibling('p')
            desc_text = desc_p.get_text()
            while desc_p.next_sibling and desc_p.next_sibling.name != 'h3':
                # print(f'next desc_p.next_sibling.name={desc_p.next_sibling.name}')
                desc_p = desc_p.next_sibling
                if desc_p.name == 'p':
                    desc_text += desc_p.get_text()
            print(f'--> SUCCESS finding episode description for episode_key={row["episode_key"]} at request_url={request_url}:')
            episodes_df.loc[episodes_df['title'] == row['title'], descr_col] = desc_text.replace('\n', ' ')
            successful += 1
        
        print(f'Finished desc load for description_source={description_source}, successful={successful} failed={failed}')

    elif description_source == 'imdb':
        pass


JOHANW_TITLE_VARIATIONS = {
    'FIRST BORN': 'FIRSTBORN', 'DESCENT': 'DESCENT, PART I', 'INNER LIGHT': 'THE INNER LIGHT', 'FISTFUL OF DATAS': 'A FISTFUL OF DATAS',
    'THE BEST OF BOTH WORLDS': 'THE BEST OF BOTH WORLDS, PART I', 'BEST OF BOTH WORLDS, PART II': 'THE BEST OF BOTH WORLDS, PART II',
    'DATA`S DAY': 'DATA\'S DAY', 'HALF-LIFE': 'HALF A LIFE', 'REDEMPTION': 'REDEMPTION, PART I', 'REDEMPTION II': 'REDEMPTION, PART II',
    'UNIFICATION I': 'UNIFICATION, PART I', 'UNIFICATION II': 'UNIFICATION, PART II', 'THE COST OF LIVING': 'COST OF LIVING',
    'INNER LIGHT': 'THE INNER LIGHT', 'TIME\'S ARROW': 'TIME\'S ARROW, PART I', 'DESCENT': 'DESCENT, PART I', 
    'All GOOD THINGS, PART I': 'ALL GOOD THINGS...', 'ALL GOOD THINGS, PART II': 'ALL GOOD THINGS...'
}


MEMORY_ALPHA_TITLE_VARIATIONS = {
    'Sins of the Father': 'Sins of The Father',
    'I, Borg': 'I Borg'
}
    

DESCRIPTION_SOURCES = {
    'memory_alpha': 'https://memory-alpha.fandom.com/wiki/',
    'johanw': 'https://johanw.home.xs4all.nl/sttng.html',
    'imdb': 'TODO'
}


if __name__ == '__main__':
    main()
