import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from app.es.es_metadata import VALID_ES_INDEXES
import app.es.es_query_builder as esqb


def main():
    '''
    Run this to explicitly define index mappings anytime an index is blown away. Not doing so will result in an index being auto-created with the wrong
    auto-assigned data types, breaking query functionality down the line.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", "-i", help="Index name", required=False)
    args = parser.parse_args()
    index_name = None
    if args.index_name:
        index_name = args.index_name
    print(f'Begin init_es script for index_name={index_name}')

    if index_name:
        if index_name not in VALID_ES_INDEXES:
            return {"error": f"Failed to initialize index_name=`{index_name}`, valid_indexes={VALID_ES_INDEXES}"}
        if index_name == 'transcripts':
            esqb.init_transcripts_index()
        elif index_name == 'narratives':
            esqb.init_narratives_index()
        elif index_name == 'speakers':
            esqb.init_speakers_index()
        elif index_name == 'speaker_seasons':
            esqb.init_speaker_seasons_index()
        elif index_name == 'speaker_episodes':
            esqb.init_speaker_episodes_index()
        elif index_name == 'speaker_embeddings_unified':
            esqb.init_speaker_unified_index()
        elif index_name == 'topics':
            esqb.init_topics_index()
        elif index_name == 'episode_topics':
            esqb.init_episode_topics_index()
        elif index_name == 'speaker_topics':
            esqb.init_speaker_topics_index()
        elif index_name == 'speaker_season_topics':
            esqb.init_speaker_season_topics_index()
        elif index_name == 'speaker_episode_topics':
            esqb.init_speaker_episode_topics_index()
        initialized_indexes = [index_name]
    else:
        print(f'in init_es index_name not set so all indexes initd')
        esqb.init_transcripts_index()
        esqb.init_narratives_index()
        esqb.init_speakers_index()
        esqb.init_speaker_seasons_index()
        esqb.init_speaker_episodes_index()
        esqb.init_speaker_unified_index()
        esqb.init_topics_index()
        esqb.init_episode_topics_index()
        esqb.init_speaker_topics_index()
        esqb.init_speaker_season_topics_index()
        esqb.init_speaker_episode_topics_index()
        initialized_indexes = VALID_ES_INDEXES

    print(f'Successfully initialized es indices: {initialized_indexes}')

    report = {"initialized_indexes": initialized_indexes}
    print(report)
    return report


if __name__ == '__main__':
    main()
    