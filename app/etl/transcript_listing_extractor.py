from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import re

import app.database.dao as dao
from app.models import TranscriptSource, Episode
from app.show_metadata import ShowKey, show_metadata


def parse_episode_listing_soup(show_key: ShowKey, episode_listing_soup: BeautifulSoup) -> list:
    episodes = []
    season_tables = episode_listing_soup.findAll("table", {"class": "wikitable plainrowheaders wikiepisodetable"})
    season = 1
    for season_table in season_tables:
        season_df = pd.read_html(str(season_table))[0]
        normalize_column_names(season_df)
        for row_i, episode_df_row in season_df.iterrows():
            episode = init_episode_from_wiki_df(show_key, episode_df_row, season, row_i+1)
            # TODO do try/except instead?
            if episode:
                episodes.append(episode)
        season += 1
    return episodes


def normalize_column_names(season_df: pd.DataFrame) -> None:
    col_updates = {}
    for col in season_df.columns:
        # strip wikipedia annotations (text enclosed in square brackets)
        new_col = re.sub(r'\[.*\]', '', col).strip()
        if new_col != col:
            col_updates[col] = new_col
    if col_updates:
        season_df.rename(columns={k:v for k,v in col_updates.items()}, inplace=True)


def init_episode_from_wiki_df(show_key: ShowKey, episode_df_row: pd.Series, season: int, row_i: int) -> Episode:
    # skip non-season specials
    if 'No. in season' not in episode_df_row:
        return None
    episode = Episode()
    episode.show_key = show_key.value
    episode.season = season
    episode.title = episode_df_row['Title'].strip('\"')
    fmt = '%B %d, %Y'
    episode.air_date = datetime.strptime(episode_df_row['Original air date'], fmt)
    if show_key == ShowKey.GoT:
        episode.sequence_in_season = int(episode_df_row['No. in season'])
        episode.external_key = episode.title.replace(' ', '_')
    elif show_key == ShowKey.TNG:
        episode.sequence_in_season = row_i
        if 'Prod. code' in episode_df_row:
            episode.external_key = str(episode_df_row['Prod. code'])[:3]
        else:
            episode.external_key = episode.title.replace(' ', '_')

    return episode


def parse_transcript_url_listing_soup(show_key: ShowKey, listing_soup: BeautifulSoup) -> dict:
    episode_transcripts_by_type = {}

    a_tags = [a_tag for a_tag in listing_soup.find_all('a')]
    all_urls = [a_tag.get('href') for a_tag in a_tags if a_tag.get('href')]

    # for tx_type, tx_string in show_metadata[show_key]['transcript_types'].items():
    # urls_already_added = []
    for tx_string in show_metadata[show_key]['transcript_type_match_strings']:
        epidose_keys_to_transcript_urls = extract_episode_keys_and_transcript_urls(show_key, all_urls, tx_string)
        print(f'for tx_string={tx_string} initial len(epidose_keys_to_transcript_urls)={len(epidose_keys_to_transcript_urls)}')
        # for external_key, tx_url in epidose_keys_to_transcript_urls.items():
        #     if tx_url in urls_already_added:
        #         del epidose_keys_to_transcript_urls[external_key]
        #     else:
        #         urls_already_added.append(tx_url)
        # print(f'for tx_string={tx_string} final len(epidose_keys_to_transcript_urls)={len(epidose_keys_to_transcript_urls)}')
        episode_transcripts_by_type[tx_string] = epidose_keys_to_transcript_urls
    
    return episode_transcripts_by_type


def extract_episode_keys_and_transcript_urls(show_key: ShowKey, all_urls: list, transcript_type_string_match: str) -> dict:
    show_transcripts_domain = show_metadata[show_key]['show_transcripts_domain']
    epidose_keys_to_transcript_urls = {}
    if show_key == "GoT":
        transcript_urls = [show_transcripts_domain + url for url in all_urls if transcript_type_string_match in url]
        for tx_url in transcript_urls:
            episode_key = tx_url.removeprefix(show_metadata[show_key]['show_transcripts_domain'])
            episode_key = episode_key.removeprefix(show_metadata[show_key]['episode_subdir'])
            episode_key = episode_key.replace(transcript_type_string_match, '')
            epidose_keys_to_transcript_urls[episode_key] = tx_url

    elif show_key == 'TNG':
        transcript_urls = [show_transcripts_domain + url for url in all_urls if url.split('.')[0].isdigit()]
        for tx_url in transcript_urls:
            transcript_file = tx_url.split('/')[-1]
            episode_key = transcript_file.split('.')[0]
            epidose_keys_to_transcript_urls[episode_key] = tx_url

    return epidose_keys_to_transcript_urls


async def match_episodes_to_transcript_urls(show_key: ShowKey, episode_transcripts_by_type: dict) -> list[TranscriptSource]:
    all_episodes = await dao.fetch_episodes(show_key.value)
    print(f'for show_key={show_key} len(all_episodes)={len(all_episodes)}')

    transcript_sources = []
    for episode in all_episodes:
        for tx_string, episode_keys_to_tx_urls in episode_transcripts_by_type.items():
            if episode.external_key in episode_keys_to_tx_urls:
                tx_source = TranscriptSource(episode=episode, transcript_type=tx_string, transcript_url=episode_keys_to_tx_urls[episode.external_key])
                transcript_sources.append(tx_source)

    return transcript_sources
    