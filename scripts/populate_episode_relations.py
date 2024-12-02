import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import app.es.es_query_builder as esqb
import app.es.es_read_router as esr
from app.nlp.nlp_metadata import ACTIVE_VENDOR_VERSIONS
from app.show_metadata import ShowKey


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--model_vendor", "-m", help="Model vendor", required=True)
    parser.add_argument("--model_version", "-v", help="Model version", required=True)
    parser.add_argument("--limit", "-l", help="Limit", required=False)
    args = parser.parse_args()
    show_key = args.show_key
    model_vendor = args.model_vendor
    model_version = args.model_version
    if args.limit:
        limit = args.limit
    else:
        limit = 30
    print(f'Begin populate_episode_relations script for show_key={show_key} model_vendor={model_vendor} model_version={model_version}')

    if (model_vendor, model_version) not in ACTIVE_VENDOR_VERSIONS and (model_vendor, model_version) != ('es','mlt'):
        print(f'invalid model_vendor:model_version combo {model_vendor}:{model_version}')
        return
    
    doc_ids = esr.fetch_doc_ids(ShowKey(show_key))
    episode_doc_ids = doc_ids['doc_ids']
    print(f'Fetched {len(episode_doc_ids)} episodes for show_key={show_key}. Begin generating and writing relations to es transcripts index.')
    
    episodes_to_relations = {}
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        print(f'Begin generating relations for episode {show_key}_{episode_key}.')
        if (model_vendor, model_version) == ('es','mlt'):
            similar_episodes = esr.more_like_this(ShowKey(show_key), episode_key)
        else:
            similar_episodes = esr.episode_mlt_vector_search(ShowKey(show_key), episode_key, model_vendor=model_vendor, model_version=model_version)
        # only keep the episode keys and corresponding scores 
        # sim_eps = [f"{sim_ep['episode_key']}|{sim_ep['score']}" for sim_ep in similar_episodes['matches']]
        episodes_to_relations[doc_id] = similar_episodes
    
    print(f'Begin writing relations for {len(episodes_to_relations)} episodes to es transcripts.')
    episodes_to_relations = esqb.populate_episode_relations(show_key, model_vendor, model_version, episodes_to_relations, limit=limit)
    print(f'Finished writing relations for {len(episodes_to_relations)} episodes to es transcripts.')

    report = {"episodes_to_relations": episodes_to_relations}
    # print(report)
    return report


if __name__ == '__main__':
    main()
