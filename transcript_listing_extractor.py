from bs4 import BeautifulSoup

from app.models import RawEpisode
from show_metadata import ShowKey, show_metadata


async def parse_episode_listing_soup(show_key: ShowKey, listing_soup: BeautifulSoup) -> list[RawEpisode]:
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


# class LinkExtractor(object):
#     def __init__(self, show_key: str, show_meta: dict, transcript_types: list|None):
#         self.show_key = show_key
#         self.show_meta = show_meta
#         if transcript_types:
#             self.transcript_types = transcript_types
#         else:
#             self.transcript_types = list(show_meta['transcript_types'].keys())

#     def extract_links(self, listing_soup: BeautifulSoup, episode: str|None):
#         transcript_types_to_urls = {}

#         if self.show_key == 'GoT':

#             if 'Fanon' in self.transcript_types:
#                 a_tags = [a_tag for a_tag in listing_soup.find_all('a')]
#                 all_urls = [a_tag.get('href') for a_tag in a_tags if a_tag.get('href')]
#                 show_transcripts_domain = show_metadata[self.show_key]['show_transcripts_domain']
#                 transcript_type_match_string = show_metadata[self.show_key]['transcript_types']['Fanon']
#                 transcript_urls = [show_transcripts_domain + url for url in all_urls if transcript_type_match_string in url]
                
#                 # fuzzy match on episode (for now), verify only one match returned
#                 if episode:
#                     transcript_urls = [url for url in transcript_urls if episode in url]
#                     if len(transcript_urls) > 1:
#                         raise Exception(f'Multiple episodes of show_key={self.show_key} and transcript_type=Fanon matched episode={episode}. Retry request with narrower name that matches only one episode.')
#                 if transcript_urls:
#                     transcript_types_to_urls['Fanon'] = transcript_urls

#             if 'TOC' in self.transcript_types:
#                 print('GoT TOC transcripts are still a TODO')

#         elif self.show_key == 'TNG':
#             a_tags = [a_tag for a_tag in listing_soup.find_all('a')]
#             all_urls = [a_tag.get('href') for a_tag in a_tags if a_tag.get('href')]
#             show_transcripts_domain = show_metadata[self.show_key]['show_transcripts_domain']
#             transcript_urls = [show_transcripts_domain + url for url in all_urls if url.split('.')[0].isdigit()]

#             # fuzzy match on episode (for now), verify only one match returned
#             if episode:
#                 transcript_urls = [url for url in transcript_urls if episode in url]
#                 if len(transcript_urls) > 1:
#                     raise Exception(f'Multiple episodes of show_key={self.show_key} and transcript_type=Fanon matched episode={episode}. Retry request with narrower name that matches only one episode.')
#             if transcript_urls:
#                 transcript_types_to_urls['default'] = transcript_urls


#         if episode:
#             if len(transcript_types_to_urls) > 1:
#                 raise Exception(f'Multiple episodes of show_key={self.show_key} with transcript_types={", ".join(transcript_types_to_urls.keys())} matched episode={episode}. Retry request with narrower name that matches only one episode.')
#             if len(transcript_types_to_urls) == 0:
#                     raise Exception(f'No episodes of show_key={self.show_key} matched episode={episode}. Exiting.')

#         print('----------------------------------------------------------------------------')
#         print(f'len(transcript_urls)={len(transcript_types_to_urls)}')
#         print('----------------------------------------------------------------------------')
#         print(f'transcript_urls={transcript_types_to_urls}')
#         return transcript_types_to_urls
    