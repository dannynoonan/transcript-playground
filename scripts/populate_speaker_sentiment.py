import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import app.data_service.sentiment_populator as sp
import app.routers.es_read_router as esr
from app.show_metadata import ShowKey
from itertools import chain


def main():
    # parse script params
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--analyzer", "-a", help="Analyzer", required=True)
    parser.add_argument("--speaker", "-c", help="Speaker", required=False)
    parser.add_argument("--min_episode_count", "-m", help="Min episode count", required=False)
    # parser.add_argument("--overwrite_csv", "-o", help="Overwrite CSV file", required=False)
    # parser.add_argument("--write_to_es", "-w", help="Write to es", required=False)
    args = parser.parse_args()

    # TODO haven't solved for setting this correctly, requires altering exit_if_unauthorized to run 
    user_dependency = None

    speaker = None
    min_episode_count = None
    # if args.speaker: 
    #     speakers = [args.speaker]
    if args.speaker: 
        speaker = args.speaker
    if args.min_episode_count: 
        min_episode_count = args.min_episode_count

    speakers_to_episode_keys = {}

    if speaker:
        # response = esr.fetch_speaker(ShowKey(args.show_key), speaker, user_dependency)
        response = esr.fetch_speaker(ShowKey(args.show_key), speaker)
        if 'speaker' not in response or 'seasons_to_episode_keys' not in response['speaker']:
            print(f'Failure to fetch speaker={speaker} for show_key={args.show_key}, `speaker` or `seasons_to_episode_keys` were not in `fetch_speaker` response')
            return
        print(f"response['speaker']['seasons_to_episode_keys']={response['speaker']['seasons_to_episode_keys']}")
        episode_keys = []
        for _, e_keys in response['speaker']['seasons_to_episode_keys'].items():
            episode_keys.extend(e_keys)
        # episode_lists = [e_keys for _,e_keys in response['speaker']['seasons_to_episode_keys']]
        # print(f'episode_keys={episode_keys}')
        speakers_to_episode_keys[speaker] = episode_keys
        
    elif min_episode_count:
        # response = esr.fetch_indexed_speakers(ShowKey(args.show_key), user_dependency, extra_fields='seasons_to_episode_keys', min_episode_count=min_episode_count)
        response = esr.fetch_indexed_speakers(ShowKey(args.show_key), extra_fields='seasons_to_episode_keys', min_episode_count=min_episode_count)
        if not 'speakers' in response:
            print(f'Failure to fetch indexed speakers for show_key={args.show_key} with min_episode_count={min_episode_count}, `speakers` was not in `fetch_indexed_speakers` response')
            return
        for speaker_data in response['speakers']:
            if 'speaker' not in speaker_data:
                print(f'Invalid speaker_data encountered in `fetch_indexed_speakers` response, skipping speaker_data={speaker_data}')
                continue
            if 'seasons_to_episode_keys' not in speaker_data:
                print(f"Failure to fetch episode_keys for speaker={speaker_data['speaker']} show_key={args.show_key}, `seasons_to_episode_keys` was not in `fetch_indexed_speakers` response. Skipping.")
                continue
            speaker_episode_keys = []
            for _, e_keys in speaker_data['seasons_to_episode_keys'].items():
                speaker_episode_keys.extend(e_keys)
            speakers_to_episode_keys[speaker_data['speaker']] = speaker_episode_keys
    else:
        print(f'Either `speaker` (-c) or `threshold` (-t) is required')
        return 
    
    print(f'speakers_to_episode_keys={speakers_to_episode_keys}')

    for speaker, episode_keys in speakers_to_episode_keys.items():
        for e_key in episode_keys:
            if e_key == '161':
                continue
            sp.copy_episode_sentiment(args.show_key, speaker, e_key, args.analyzer, aggregate=True)


if __name__ == '__main__':
    main()
