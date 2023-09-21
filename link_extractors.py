from bs4 import BeautifulSoup

import show_metadata


class LinkExtractor(object):
    def __init__(self, show_key: str, show_meta: dict, transcript_types: list|None):
        self.show_key = show_key
        self.show_meta = show_meta
        if transcript_types:
            self.transcript_types = transcript_types
        else:
            self.transcript_types = list(show_meta['transcript_types'].keys())

    def extract_links(self, listing_soup: BeautifulSoup, episode: str|None):
        transcript_types_to_urls = {}

        if self.show_key == 'GoT':

            if 'Fanon' in self.transcript_types:
                a_tags = [a_tag for a_tag in listing_soup.find_all('a')]
                all_urls = [a_tag.get('href') for a_tag in a_tags if a_tag.get('href')]
                show_transcripts_domain = show_metadata.shows[self.show_key]['show_transcripts_domain']
                transcript_type_match_string = show_metadata.shows[self.show_key]['transcript_types']['Fanon']
                transcript_urls = [show_transcripts_domain + url for url in all_urls if transcript_type_match_string in url]
                
                # fuzzy match on episode (for now), verify only one match returned
                if episode:
                    transcript_urls = [url for url in transcript_urls if episode in url]
                    if len(transcript_urls) > 1:
                        raise Exception(f'Multiple episodes of show_key={self.show_key} and transcript_type=Fanon matched episode={episode}. Retry request with narrower name that matches only one episode.')
                if transcript_urls:
                    transcript_types_to_urls['Fanon'] = transcript_urls

            if 'TOC' in self.transcript_types:
                print('GoT TOC transcripts are still a TODO')

        elif self.show_key == 'TNG':
            a_tags = [a_tag for a_tag in listing_soup.find_all('a')]
            all_urls = [a_tag.get('href') for a_tag in a_tags if a_tag.get('href')]
            show_transcripts_domain = show_metadata.shows[self.show_key]['show_transcripts_domain']
            transcript_urls = [show_transcripts_domain + url for url in all_urls if url.split('.')[0].isdigit()]

            # fuzzy match on episode (for now), verify only one match returned
            if episode:
                transcript_urls = [url for url in transcript_urls if episode in url]
                if len(transcript_urls) > 1:
                    raise Exception(f'Multiple episodes of show_key={self.show_key} and transcript_type=Fanon matched episode={episode}. Retry request with narrower name that matches only one episode.')
            if transcript_urls:
                transcript_types_to_urls['default'] = transcript_urls


        if episode:
            if len(transcript_types_to_urls) > 1:
                raise Exception(f'Multiple episodes of show_key={self.show_key} with transcript_types={", ".join(transcript_types_to_urls.keys())} matched episode={episode}. Retry request with narrower name that matches only one episode.')
            if len(transcript_types_to_urls) == 0:
                    raise Exception(f'No episodes of show_key={self.show_key} matched episode={episode}. Exiting.')

        print('----------------------------------------------------------------------------')
        print(f'len(transcript_urls)={len(transcript_types_to_urls)}')
        print('----------------------------------------------------------------------------')
        print(f'transcript_urls={transcript_types_to_urls}')
        return transcript_types_to_urls
    