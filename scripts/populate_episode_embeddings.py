import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import app.routers.es_read_router as esr
import app.routers.es_write_router as esw
from app.show_metadata import ShowKey


def main():
    '''
    Bulk run of `/esw/populate_episode_embeddings` for all episodes of a given show
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--model_vendor", "-m", help="Model vendor", required=True)
    parser.add_argument("--model_version", "-v", help="Model version", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    model_vendor = args.model_vendor
    model_version = args.model_version
    print(f'Begin populate_episode_embeddings script for show_key={show_key} model_vendor={model_vendor} model_version={model_version}')

    # TODO haven't solved for setting this correctly, requires altering exit_if_unauthorized to run 
    user_dependency = None

    doc_ids = esr.fetch_doc_ids(ShowKey(show_key), user_dependency)
    episode_doc_ids = doc_ids['doc_ids']
    print(f'Fetched {len(episode_doc_ids)} episodes for show_key={show_key}. Begin generating and writing embeddings to es transcripts index.')

    processed_episode_keys = []
    failed_episode_keys = []
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        print(f'Begin populate_episode_embeddings for episode {show_key}_{episode_key}.')
        try:
            esw.populate_episode_embeddings(ShowKey(show_key), episode_key, model_vendor, model_version, user_dependency)
            processed_episode_keys.append(episode_key)
        except Exception as e:
            print(f'Failed to populate_episode_embeddings for episode {show_key}_{episode_key}: {e}')
            failed_episode_keys.append(episode_key)

    print(f'Finished generating and writing embeddings to es transcripts index for {len(processed_episode_keys)} of {len(episode_doc_ids)} episodes.')

    report = {"processed_episode_keys": processed_episode_keys, "failed_episode_keys": failed_episode_keys}
    print(report)
    return report


if __name__ == '__main__':
    main()
