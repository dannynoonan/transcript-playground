import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import app.es.es_read_router as esr
import app.es.es_write_router as esw
from app.show_metadata import ShowKey


def main():
    '''
    Generate and populate all narrative sequences for a show
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    print(f'Begin populate_episode_narratives script for show_key={show_key}')

    doc_ids_response = esr.fetch_doc_ids(ShowKey(show_key))
    doc_ids = doc_ids_response['doc_ids']
    print(f'Fetched {len(doc_ids)} episodes for show_key={show_key}. Begin generating and writing episode narratives to es.')

    successful_keys = []
    failed_keys = []
    for doc_id in doc_ids:
        print(f'Begin populating narratives for episode {doc_id}')
        episode_key = doc_id.split('_')[1]
        narrative_sequences_response = esw.populate_episode_narratives(ShowKey(show_key), episode_key)
        if 'narrative_sequences' not in narrative_sequences_response:
            print(f'Failed to populate_episode_narratives for show_key={show_key} episode_key={episode_key}')
            failed_keys.append(episode_key)
        else:
            successful_keys.append(episode_key)

    print(f'Finished populate_episode_narratives script, {successful_keys} of {len(doc_ids)} populated successfully.')

    report = {"successful_keys": successful_keys, "failed_keys": failed_keys}
    print(report)
    return report


if __name__ == '__main__':
    main()
