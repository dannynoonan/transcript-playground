#!/usr/bin/env python
import argparse
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen

import show_metadata
from link_extractors import LinkExtractor
from transcript_extractors import TranscriptExtractor


def parse_args():
    # init parser and recognized args
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", "-s", help="Full name or shorthand name of show to be loaded", required=True)
    parser.add_argument("--transcript_type", "-t", help="Specific transcript format type to be loaded, or ALL transcript format types (default)")
    parser.add_argument("--episode", "-e", help="A specific episode_id / episode_title, or ALL episodes (default)")
    # read cmd-line args
    return parser.parse_args()


def parse_show(show_arg: str):
    show_key = None
    show_meta = None
    if not show_arg:
        raise Exception("Transcript import request error: 'show' arg is required")
    if show_arg in show_metadata.shows.keys():
        show_key = show_arg
        show_meta = show_metadata.shows[show_key]
    else:
        for s_key, s_meta in show_metadata.shows.items():
            if show_arg == s_meta['full_name']:
                show_key = s_key
                show_meta = s_meta
                break
    if not show_key:
        raise Exception(f"Transcript import request error: no match for show={args.show}")
    return show_key, show_meta


def parse_transcript_type(transcript_type_arg: str|None, show_key: str, show_meta: dict):
    transcript_types = []
    if not transcript_type_arg or transcript_type_arg == 'ALL':
        transcript_types = show_meta['transcript_types'].keys()
        print(f"--> Transcript import request will cover all transcript_types for show_key={show_key}")
    elif transcript_type_arg in show_meta['transcript_types'].keys():
        transcript_types = [transcript_type_arg]
        print(f"--> Transcript import request will be limited to transcript_type={transcript_types[0]} for show_key={show_key}")

    else:
        raise Exception(f"Transcript import request error: no match for transcript_type={transcript_type_arg} for show_key={show_key}")
    return transcript_types


def parse_episode(episode_arg: str|None, show_key: str, transcript_types: list):
    episode = None
    if not episode_arg or episode_arg == 'ALL':
        # no-op
        print(f"--> Transcript import request will cover all episodes for show_key={show_key} and transcript_types={', '.join(transcript_types)}")
    else:
        episode = episode_arg
        print(f"--> Transcript import request will only operate on episode={episode} for show_key={show_key}")
    return episode


def main():
    args = parse_args()

    print(f"Received request for: show={args.show}, transcript_type={args.transcript_type}, episode={args.episode}\n")

    # show
    show_key, show_meta = parse_show(args.show)

    # transcript_type
    transcript_types = parse_transcript_type(args.transcript_type, show_key, show_meta)
    
    # episode
    episode = parse_episode(args.episode, show_key, transcript_types)

    print('----------------------------------------------------------------------------')
    print(f'show_key={show_key}, transcript_types={", ".join(transcript_types)}, episode={episode}')

    # soupify page with links to transcripts 
    show_transcripts_domain = show_metadata.shows[show_key]['show_transcripts_domain']
    listing_url = show_metadata.shows[show_key]['listing_url']
    listing_html = requests.get(show_transcripts_domain + listing_url)
    listing_soup = BeautifulSoup(listing_html.text, 'html.parser')

    # extract transcript links from soupified html
    link_extractor = LinkExtractor(show_key, show_meta, transcript_types)
    transcript_types_to_urls = link_extractor.extract_links(listing_soup, episode)

    # extract each transcript
    for transcript_type, transcript_urls in transcript_types_to_urls.items():
        for transcript_url in transcript_urls:
            print('----------------------------------------------------------------------------')
            print(f'Begin processing transcript_url={transcript_url}')

            transcript_extractor = TranscriptExtractor(show_key, transcript_type)
            transcript_response = requests.get(transcript_url)
            transcript_soup = BeautifulSoup(transcript_response.content.decode('utf-8'), 'html.parser')
            episode = transcript_extractor.extract_transcript(transcript_soup)

            # extract title from url (temporary hack)
            episode.transcript_url = transcript_url
            title = transcript_url.removeprefix(show_transcripts_domain)
            title = title.removeprefix(show_meta['episode_subdir'])
            transcript_type_string_match = show_meta['transcript_types'][transcript_type]
            title = title.replace(transcript_type_string_match, '')
            episode.title = title

            # json_episode = json.dumps(episode.toJSON())
            # print(episode.toJSON())

            file_path = f'shows/{show_key}/{episode.title}.txt'

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(episode.toJSON())

            print(f'Finished processing transcript_url={transcript_url}, wrote to file_path={file_path}')

    print('----------------------------------------------------------------------------')
    print('Transcript import complete.')
    print('----------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
