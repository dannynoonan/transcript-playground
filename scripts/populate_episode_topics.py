import argparse
from operator import itemgetter
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import app.data_service.topicfidf_calculator as tfcalc
import app.data_service.field_flattener as fflat
import app.es.es_query_builder as esqb
import app.es.es_read_router as esr
import app.es.es_write_router as esw
from app.show_metadata import ShowKey


def main():
    '''
    For specified topic_grouping, generate and store topic mappings for all series episodes,
    then calculate 'tfidf'-like scores for all episode_topics and store in `tfidf_score` field
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--topic_grouping", "-g", help="Topic grouping", required=True)
    parser.add_argument("--model_vendor", "-m", help="Model vendor", required=True)
    parser.add_argument("--model_version", "-v", help="Model version", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    topic_grouping = args.topic_grouping
    model_vendor = args.model_vendor
    model_version = args.model_version
    print(f'Begin populate_episode_topics script for show_key={show_key} topic_grouping={topic_grouping} model_vendor={model_vendor} model_version={model_version}')

    doc_ids = esr.fetch_doc_ids(ShowKey(show_key))
    episode_doc_ids = doc_ids['doc_ids']
    print(f'Fetched {len(episode_doc_ids)} episodes for show_key={show_key}. Begin vector similarity mapping to topic_grouping={topic_grouping}.')

    processed_episode_keys = []
    failed_episode_keys = []
    for doc_id in episode_doc_ids:
        episode_key = doc_id.split('_')[-1]
        print(f'Begin populating episode topics for episode {show_key}_{episode_key}.')
        try:
            esw.populate_episode_topics(ShowKey(show_key), episode_key, topic_grouping, model_vendor, model_version)
            processed_episode_keys.append(episode_key)
        except Exception as e:
            print(f'Failed to populate_episode_topics for episode_key={episode_key}: {e}')
            failed_episode_keys.append(episode_key)

    print(f'Finished vector similarity mapping and es writing for {len(processed_episode_keys)} out of {len(episode_doc_ids)} episodes to topic_grouping={topic_grouping}.')

    print(f"Begin calculating 'tfidf-like' scores for all episode_topics")
    ekey_tkey_scores, topic_idfs = tfcalc.calculate_topic_freq_idf(ShowKey(show_key), topic_grouping, model_vendor, model_version)

    successful_episode_keys = []
    # for each episode, generate and store "tf-idf" values per topic
    for e_key, t_keys_to_scores in ekey_tkey_scores.items():

        print(f'Processing episode_topics for episode_key={e_key}')
        episode_topics = []
        for t_key, score in t_keys_to_scores.items():
            episode_topic = tfcalc.set_episode_topic_tfidf(ShowKey(show_key), t_key, e_key, score, topic_idfs, topic_grouping, model_vendor, model_version)
            episode_topics.append(episode_topic)
            
        # save simplified subset of episode_topics to es_episode.topics_X_tfidf
        # TODO topics_universal should either be topics_universal_{model_name} or a dict keying off of model_name instead of a list of topics
        # For now I'm only indexing topics generated with openai:3small embeddings 
        if topic_grouping in ['universalGenres'] and model_vendor == 'openai' and model_version == '3small':
            print(f'model {model_vendor}:{model_version} is also indexed in tfidf_score field for episode_key={e_key}')
            tfidf_sorted_episode_topics = sorted(episode_topics, key=itemgetter('tfidf_score'), reverse=True)
            simple_episode_topics = fflat.flatten_es_topics(tfidf_sorted_episode_topics)
            esw.populate_episode_tfidf_topics(ShowKey(show_key), e_key, topic_grouping, simple_episode_topics)
            successful_episode_keys.append(e_key)
        else:
            print(f'model {model_vendor}:{model_version} is not indexed in tfidf_score field, skipping for episode_key={e_key}')
            successful_episode_keys.append(e_key)

    print(f"Finished calculating and persisting 'tfidf-like' topic scores for show_key={show_key} topic_grouping={topic_grouping} model_vendor={model_vendor} model_version={model_version}")

    report = {"attempted": len(ekey_tkey_scores), "successful": len(successful_episode_keys), "successful_episode_keys": successful_episode_keys}
    print(report)
    return report


if __name__ == '__main__':
    main()
