import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import app.es.es_read_router as esr
import app.es.es_write_router as esw
from app.show_metadata import SPEAKERS_TO_IGNORE, ShowKey


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    print(f'Begin index_speakers script for show_key={show_key}')

    response = esr.agg_episodes_by_speaker(ShowKey(show_key))
    speaker_episode_counts = response['episodes_by_speaker']
    valid_speakers = [s for s,_ in speaker_episode_counts.items() if '+' not in s and s not in SPEAKERS_TO_IGNORE]
    print(f'Fetched {len(speaker_episode_counts)} speakers for show_key {show_key}, trimmed to {len(valid_speakers)} using SPEAKERS_TO_IGNORE. Begin writing to es speakers index.')

    attempt_count = 0
    successful = []
    failed = []
    for speaker in valid_speakers:
        attempt_count += 1
        print(f'Begin indexing speaker {speaker}')
        try:
            response = esw.index_speaker(ShowKey(show_key), speaker)
            if "speaker" in response:
                print(f"Successfully indexed speaker={speaker}")
                successful.append(speaker)
            else:
                print(f"Failed to index speaker={speaker}: {response['Error']}")
                failed.append(speaker)
        except Exception as e:
            print(f"Failed to index speaker={speaker}: {e}")
            failed.append(speaker)

    print(f'Finished loading {attempt_count} out of {len(valid_speakers)} speakers')

    report = {"attempt_count": attempt_count, "successful": successful, "failed": failed}
    print(report)
    return report


if __name__ == '__main__':
    main()
