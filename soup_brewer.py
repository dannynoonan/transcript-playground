#!/usr/bin/env python
from bs4 import BeautifulSoup
import requests

from app.models import RawEpisode
from show_metadata import show_metadata


async def get_episode_listing_soup(show_key: str):
    # soupify page with links to transcripts 
    show_transcripts_domain = show_metadata[show_key]['show_transcripts_domain']
    listing_url = show_metadata[show_key]['listing_url']
    listing_html = requests.get(show_transcripts_domain + listing_url)
    listing_soup = BeautifulSoup(listing_html.text, 'html.parser')
    return listing_soup


async def get_transcript_soup(raw_episode: RawEpisode) -> BeautifulSoup:
    print(f'Begin importing raw_episode={raw_episode} with transcript_url={raw_episode.transcript_url}')
    transcript_response = requests.get(raw_episode.transcript_url)
    transcript_soup = BeautifulSoup(transcript_response.content.decode('utf-8'), 'html.parser')
    return transcript_soup
