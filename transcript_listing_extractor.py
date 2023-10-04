from bs4 import BeautifulSoup
import pandas as pd

from app.models import RawEpisode, Episode
from show_metadata import ShowKey, show_metadata


async def parse_episode_listing_soup(show_key: ShowKey, episode_listing_soup: BeautifulSoup):
    episodes = []
    season_tables = episode_listing_soup.findAll("table", {"class": "wikitable plainrowheaders wikiepisodetable"})
    season = 1
    for season_table in season_tables:
        season_df = pd.read_html(str(season_table))[0]
        # print(f'for season={season} season_df.columns={season_df.columns}')
        for index, episode_df_row in season_df.iterrows():
            # skip non-season specials
            if 'No. inseason' not in season_df.columns:
                continue
            episode = await init_episode_from_wiki_df(show_key, episode_df_row, season)
            episodes.append(episode)
        season += 1
    return episodes


async def init_episode_from_wiki_df(show_key: ShowKey, episode_df_row: pd.Series, season: int) -> Episode:
    episode = Episode()
    episode.show_key = show_key.value
    episode.season = season
    episode.title = episode_df_row['Title'].strip('\"')
    episode.sequence_in_season = int(episode_df_row['No. inseason'])
    # episode.air_date = episode_df_row['Original air date']
    if show_key == ShowKey.GoT:
        episode.external_key = episode.title.replace(' ', '_')
    elif show_key == ShowKey.TNG:
        if 'Prod. code' in episode_df_row:
            episode.external_key = episode_df_row['Prod. code']
        else:
            episode.external_key = 'TODO'

    return episode


async def parse_transcript_url_listing_soup(show_key: ShowKey, listing_soup: BeautifulSoup) -> list[RawEpisode]:
    raw_episodes = []

    a_tags = [a_tag for a_tag in listing_soup.find_all('a')]
    all_urls = [a_tag.get('href') for a_tag in a_tags if a_tag.get('href')]

    # for tx_type, tx_string in show_metadata[show_key]['transcript_types'].items():
    urls_already_added = []
    for tx_string in show_metadata[show_key]['transcript_type_match_strings']:
        epidose_keys_to_transcript_urls = await extract_episode_keys_and_transcript_urls(show_key, all_urls, tx_string)
        for external_key, tx_url in epidose_keys_to_transcript_urls.items():
            if tx_url not in urls_already_added:
                urls_already_added.append(tx_url)
                raw_episodes.append(RawEpisode(show_key=show_key.value, external_key=external_key, transcript_type=tx_string, transcript_url=tx_url))
    
    print(f'len(raw_episodes)={len(raw_episodes)}')
    print(f'raw_episodes={raw_episodes}')
    return raw_episodes


async def extract_episode_keys_and_transcript_urls(show_key: ShowKey, all_urls: list, transcript_type_string_match: str) -> dict:
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

    