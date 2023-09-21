#!/usr/bin/env python
from bs4 import BeautifulSoup
import json
import requests
from urllib.request import urlopen

import show_metadata
from link_extractors import LinkExtractor
from transcript_extractors import TranscriptExtractor


def get_show_meta(show_key: str):
    show_meta = None
    if show_key in show_metadata.shows.keys():
        show_meta = show_metadata.shows[show_key]
    else:
        raise Exception(f"No show_metadata match for show_key={show_key}")
    return show_meta

def get_show_listing_soup(show_key: str):
    # soupify page with links to transcripts 
    show_transcripts_domain = show_metadata.shows[show_key]['show_transcripts_domain']
    listing_url = show_metadata.shows[show_key]['listing_url']
    listing_html = requests.get(show_transcripts_domain + listing_url)
    listing_soup = BeautifulSoup(listing_html.text, 'html.parser')
    return listing_soup


def import_transcript_by_episode_key(show_key: str, episode_key: str):
    print(f'import_transcript: show_key={show_key}, episode_key={episode_key}')
    show_meta = get_show_meta(show_key)
    listing_soup = get_show_listing_soup(show_key)

    # extract transcript links from soupified html
    link_extractor = LinkExtractor(show_key, show_meta, None)
    transcript_types_to_urls = link_extractor.extract_links(listing_soup, episode_key)

    return import_transcripts(show_key, show_meta, transcript_types_to_urls)


def import_transcripts_by_type(show_key: str, transcript_type: str):
    print(f'import_transcript: show_key={show_key}, transcript_type={transcript_type}')
    show_meta = get_show_meta(show_key)
    listing_soup = get_show_listing_soup(show_key)

    # extract transcript links from soupified html
    link_extractor = LinkExtractor(show_key, show_meta, [transcript_type])
    transcript_types_to_urls = link_extractor.extract_links(listing_soup, None)
    print(f'transcript_types_to_urls={transcript_types_to_urls}')

    return import_transcripts(show_key, show_meta, transcript_types_to_urls)


def import_transcripts(show_key: str, show_meta: dict, transcript_types_to_urls: dict):
    json_episodes = {}

    # extract each transcript
    for transcript_type, transcript_urls in transcript_types_to_urls.items():
        for transcript_url in transcript_urls:
            print('----------------------------------------------------------------------------')
            print(f'Begin processing transcript_url={transcript_url}')

            transcript_extractor = TranscriptExtractor(show_key, show_meta, transcript_type, transcript_url)
            transcript_response = requests.get(transcript_url)
            transcript_soup = BeautifulSoup(transcript_response.content.decode('utf-8'), 'html.parser')
            episode = transcript_extractor.extract_transcript(transcript_soup)

            # json_episodes.append(json.dumps(episode.toJSON()))
            # json_episodes.append(episode)
            # json_episodes.append(episode.toJSON())
            json_episodes[episode.external_id] = episode.toJSON()

            print(episode.toJSON())

            file_path = f'shows/{show_key}/{episode.title}.txt'

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(episode.toJSON())

            print(f'Finished processing transcript_url={transcript_url}, wrote to file_path={file_path}')

    print('----------------------------------------------------------------------------')
    print('Transcript import complete.')
    print('----------------------------------------------------------------------------')

    return json_episodes
