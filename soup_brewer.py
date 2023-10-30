#!/usr/bin/env python
from bs4 import BeautifulSoup
import requests

from app.models import TranscriptSource
from show_metadata import show_metadata, WIKIPEDIA_DOMAIN


async def get_episode_detail_listing_soup(show_key: str) -> BeautifulSoup:
    # soupify wikipedia episode listing 
    episode_listing_detail_html = requests.get(WIKIPEDIA_DOMAIN + show_metadata[show_key]['wikipedia_label'])
    return BeautifulSoup(episode_listing_detail_html.text, 'html.parser')


async def get_transcript_url_listing_soup(show_key: str) -> BeautifulSoup:
    # soupify page with links to transcripts 
    show_transcripts_domain = show_metadata[show_key]['show_transcripts_domain']
    listing_url = show_metadata[show_key]['listing_url']
    listing_html = requests.get(show_transcripts_domain + listing_url)
    return BeautifulSoup(listing_html.text, 'html.parser')


async def get_transcript_soup(transcript_source: TranscriptSource) -> BeautifulSoup:
    # soupify transcipt page
    print(f'Begin importing transcript for transcript_source={transcript_source} with transcript_url={transcript_source.transcript_url}')
    transcript_response = requests.get(transcript_source.transcript_url)
    return BeautifulSoup(transcript_response.content.decode('utf-8'), 'html.parser')
