#!/usr/bin/env python
import argparse
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen

from show_metadata import show_metadata
from link_extractors import LinkExtractor
from transcript_importer import scrape_transcript
from transcript_extractors import TranscriptExtractor


def parse_args():
    # init parser and recognized args
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", "-s", help="Shorthand name of show to be loaded", required=True)
    # parser.add_argument("--transcript_type", "-t", help="Specific transcript format type to be loaded, or ALL transcript format types (default)")
    parser.add_argument("--episode", "-e", help="A specific episode_id / episode_title, or ALL episodes (default)", required=True)
    # read cmd-line args
    return parser.parse_args()


# def parse_show(show_arg: str):
#     show_key = None
#     show_meta = None
#     if not show_arg:
#         raise Exception("Transcript import request error: 'show' arg is required")
#     if show_arg in show_metadata.keys():
#         show_key = show_arg
#         show_meta = show_metadata[show_key]
#     else:
#         for s_key, s_meta in show_metadata.items():
#             if show_arg == s_meta['full_name']:
#                 show_key = s_key
#                 show_meta = s_meta
#                 break
#     if not show_key:
#         raise Exception(f"Transcript import request error: no match for show={show_arg}")
#     return show_key, show_meta


# def parse_transcript_type(transcript_type_arg: str|None, show_key: str, show_meta: dict):
#     transcript_types = []
#     if not transcript_type_arg or transcript_type_arg == 'ALL':
#         transcript_types = show_meta['transcript_types'].keys()
#         print(f"--> Transcript import request will cover all transcript_types for show_key={show_key}")
#     elif transcript_type_arg in show_meta['transcript_types'].keys():
#         transcript_types = [transcript_type_arg]
#         print(f"--> Transcript import request will be limited to transcript_type={transcript_types[0]} for show_key={show_key}")

#     else:
#         raise Exception(f"Transcript import request error: no match for transcript_type={transcript_type_arg} for show_key={show_key}")
#     return transcript_types


# def parse_episode(episode_arg: str|None, show_key: str, transcript_types: list):
#     episode = None
#     if not episode_arg or episode_arg == 'ALL':
#         # no-op
#         print(f"--> Transcript import request will cover all episodes for show_key={show_key} and transcript_types={', '.join(transcript_types)}")
#     else:
#         episode = episode_arg
#         print(f"--> Transcript import request will only operate on episode={episode} for show_key={show_key}")
#     return episode


def main():
    args = parse_args()
    print(f"Received request for: show={args.show}, episode={args.episode}")

    if args.show not in show_metadata.keys():
        raise Exception(f"Transcript import request error: no match for show={args.show}")
        
    episode, scenes, scenes_to_events = scrape_transcript(args.show, args.episode)

    print(f'episode={episode}')
    print(f'scenes={scenes}')
    print(f'scenes_to_events={scenes_to_events}')


if __name__ == '__main__':
    main()
    