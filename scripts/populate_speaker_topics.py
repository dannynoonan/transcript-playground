import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.es.es_write_router as esw
from app.show_metadata import ShowKey


def main():
    '''
    Map speakers to topics (using knn vector cosine similarity) for all indexed speakers for a show 
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
    print(f'Begin populate_speaker_topics script for show_key={show_key} topic_grouping={topic_grouping} model_vendor={model_vendor} model_version={model_version}')

    # TODO again, not sure what's what with inconsistent usage of esr vs esqb/esrt
    s = esqb.fetch_indexed_speakers(show_key, return_fields=['speaker'])
    matches = esrt.return_speakers(s)
    if not matches:
        print(f'Unable to populate_speaker_topics: Failed to fetch_indexed_speakers for show_key={show_key}.')
        return {"error": f"Failed to fetch_indexed_speakers for show_key={show_key}"}
    
    speakers = [m['speaker'] for m in matches]
    attempt_count = 0
    success_count = 0
    successful_speakers = []
    failure_count = 0
    failed_speakers = []
    for speaker in speakers:
        attempt_count += 1
        print(f'Begin populating topics for speaker={speaker}')
        try:
            response = esw.populate_speaker_topics(ShowKey(show_key), speaker, topic_grouping, model_vendor, model_version)
            if "error" in response:
                print(f"Failed to populate_speaker_topics for speaker={speaker}: {response['error']}")
                failed_speakers.append(speaker)
                failure_count += 1
            else:
                successful_speakers.append(speaker)
                success_count += 1
        except Exception as e:
            print(f"Failed to populate_speaker_topics for speaker={speaker}: {e}")
            failed_speakers.append(speaker)
            failure_count += 1

    print(f'Finished populate_speaker_topics for show_key={show_key} topic_grouping={topic_grouping}, {success_count} of {attempt_count} populated successfully')

    report = {"attempt_count": attempt_count, "success_count": success_count, "failure_count": failure_count,
            "successful_speakers": successful_speakers, "failed_speakers": failed_speakers}
    print(report)
    return report


if __name__ == '__main__':
    main()
