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
    Generate vector embedding for all indexed speakers for a show using pre-trained Transformer models
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--model_vendor", "-m", help="Model vendor", required=True)
    parser.add_argument("--model_version", "-v", help="Model version", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    model_vendor = args.model_vendor
    model_version = args.model_version
    print(f'Begin populate_speaker_embeddings script for show_key={show_key} model_vendor={model_vendor} model_version={model_version}')

    # NOTE not sure why this one uses esqb/esrt directly and other batch indexers use esr 
    s = esqb.fetch_indexed_speakers(show_key, return_fields=['speaker'])
    matches = esrt.return_speakers(s)
    if not matches:
        print(f'Failed to fetch_indexed_speakers for show_key={show_key}')
        return
    
    print(f'Fetched {len(matches)} speakers for show_key={show_key}. Begin generating and writing embeddings to es speaker* indices.')
    speakers = [m['speaker'] for m in matches]
    request_count = 0
    success_count = 0
    skipped_count = 0
    failure_count = 0
    super_fails = []
    speaker_responses = {}
    for speaker in speakers:
        try:
            response = esw.populate_speaker_embeddings(ShowKey(show_key), speaker, model_vendor, model_version)
            speaker_responses[speaker] = response
            request_count += response['attempted_count']
            success_count += len(response['successful'])
            skipped_count += len(response['skipped'])
            failure_count += len(response['failed'])
        except Exception as e:
            print(f"Failed to populate_speaker_embeddings for speaker={speaker}: {e}")
            super_fails.append(speaker)

    print(f'Finished generating and writing embeddings to es speaker* indices for {success_count} of {request_count} speakers.')

    report = {"request_count": request_count, "success_count": success_count, "skipped_count": skipped_count, "failure_count": failure_count,
            "super_fails": super_fails, "speaker_responses": speaker_responses}
    print(report)
    return report


if __name__ == '__main__':
    main()
