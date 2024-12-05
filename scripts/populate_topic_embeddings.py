import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import app.es.es_read_router as esr
import app.es.es_write_router as esw


def main():
    '''
    Generate vector embedding for all topics in topic_grouping using pre-trained Word2Vec and Transformer models
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic_grouping", "-g", help="Topic grouping", required=True)
    parser.add_argument("--model_vendor", "-m", help="Model vendor", required=True)
    parser.add_argument("--model_version", "-v", help="Model version", required=True)
    args = parser.parse_args()
    topic_grouping = args.topic_grouping
    model_vendor = args.model_vendor
    model_version = args.model_version
    print(f'Begin populate_topic_embeddings script for topic_grouping={topic_grouping} model_vendor={model_vendor} model_version={model_version}')

    topic_grouping_response = esr.fetch_topic_grouping(topic_grouping)
    topic_keys = [t['topic_key'] for t in topic_grouping_response['topics']]
    print(f'Fetched {len(topic_keys)} topics for topic_grouping={topic_grouping}. Begin generating and writing embeddings to es topics index.')

    attempted_count = 0
    successful_topics = []
    failed_topics = []
    failure_messages = []
    for topic_key in topic_keys:
        attempted_count += 1
        print(f'Begin populate_topic_embeddings for topic_key {topic_key}')
        topic_embeddings_response = esw.populate_topic_embeddings(topic_grouping, topic_key, model_vendor, model_version)
        if 'topic' in topic_embeddings_response:
            successful_topics.append(topic_key)
        else:
            failed_topics.append(topic_key)
            if 'error' in topic_embeddings_response:
                failure_messages.append(topic_embeddings_response['error'])

    print(f'Finished generating and writing embeddings to es topics index for {len(successful_topics)} of {len(topic_keys)} episodes.')

    report = {
        'attempted_count': attempted_count, 
        'successful_topics': successful_topics, 
        'failed_topics': failed_topics, 
        'failure_messages': failure_messages
    }
    print(report)
    return report


if __name__ == '__main__':
    main()
