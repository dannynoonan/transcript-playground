#!/usr/bin/env python
import argparse
# from bs4 import BeautifulSoup
# import requests
from urllib.request import urlopen

from app.models import TranscriptSource
from show_metadata import show_metadata
from source_etl.soup_brewer import get_transcript_soup
from source_etl.transcript_extractor import parse_episode_transcript_soup


def parse_args():
    # init parser and recognized args
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", "-s", help="Shorthand name of show to be loaded", required=True)
    # parser.add_argument("--transcript_type", "-t", help="Specific transcript format type to be loaded, or ALL transcript format types (default)")
    parser.add_argument("--episode", "-e", help="A specific episode_id / episode_title, or ALL episodes (default)", required=True)
    # read cmd-line args
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Received request for: show={args.show}, episode={args.episode}")

    if args.show not in show_metadata.keys():
        raise Exception(f"Transcript import request error: no match for show={args.show}")
        
    episode_soup = get_transcript_soup(TranscriptSource(show_key=args.show, external_key=args.episode))
    # TODO this doesn't work because we no longer have transcript_type
    episode, scenes, scenes_to_events = parse_episode_transcript_soup(args.show, args.episode, None, episode_soup)

    print(f'episode={episode}')
    print(f'scenes={scenes}')
    print(f'scenes_to_events={scenes_to_events}')


if __name__ == '__main__':
    main()
    