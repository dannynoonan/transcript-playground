#!/usr/bin/env python
# from bs4 import BeautifulSoup
# import requests

# from show_metadata import show_metadata, WIKIPEDIA_DOMAIN


# async def get_episode_detail_listing_soup(show_key: str) -> BeautifulSoup:
#     # soupify wikipedia episode listing 
#     print(f'Begin get_episode_detail_listing_soup for show_key={show_key}')
#     episode_listing_detail_html = requests.get(WIKIPEDIA_DOMAIN + show_metadata[show_key]['wikipedia_label'])
#     return BeautifulSoup(episode_listing_detail_html.text, 'html.parser')


# def get_episode_listing_file_soup(episode_listing_file: str) -> BeautifulSoup:
#     # soupify episode_listing_file file
#     print(f'Begin get_episode_listing_file_soup for episode_listing_file={episode_listing_file}')
#     return BeautifulSoup(open(episode_listing_file).read())


# async def get_transcript_url_listing_soup(show_key: str) -> BeautifulSoup:
#     # soupify page with links to transcripts 
#     print(f'Begin get_transcript_url_listing_soup for show_key={show_key}')
#     show_transcripts_domain = show_metadata[show_key]['show_transcripts_domain']
#     listing_url = show_metadata[show_key]['listing_url']
#     listing_html = requests.get(show_transcripts_domain + listing_url)
#     return BeautifulSoup(listing_html.text, 'html.parser')


# def get_transcript_source_file_soup(transcript_source_file: str) -> BeautifulSoup:
#     # soupify episode_listing_file file
#     print(f'Begin get_transcript_source_file_soup for transcript_source_file={transcript_source_file}')
#     return BeautifulSoup(open(transcript_source_file).read())


# def get_transcript_soup(transcript_url: str) -> BeautifulSoup:
#     # soupify transcript page
#     print(f'Begin get_transcript_soup for transcript_url={transcript_url}')
#     transcript_html = requests.get(transcript_url)
#     return BeautifulSoup(transcript_html.content.decode('utf-8'), 'html5lib')


# def get_transcript_file_soup(transcript_file: str) -> BeautifulSoup:
#     # soupify transcript file
#     print(f'Begin get_transcript_file_soup for transcript_file={transcript_file}')
#     return BeautifulSoup(open(transcript_file).read())


# def get_episode_description_soup(url) -> BeautifulSoup:
#     # soupify episode description page
#     print(f'Begin get_episode_description_soup for url={url}')
#     episode_descriptions_html = requests.get(url)
#     return BeautifulSoup(episode_descriptions_html.content.decode('utf-8'), 'html5lib')
