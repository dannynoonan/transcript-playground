import argparse
import os
import pandas as pd
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import app.es.es_query_builder as esqb
import app.es.es_read_router as esr
from app.show_metadata import ShowKey
from app.app_metadata import BERTOPIC_DATA_DIR


def main():
    '''
    Load each bertopic_model's csv into dataframe, upsert referenced episode_narratives with mapping back to bertopic_model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    args = parser.parse_args()
    show_key = args.show_key
    # umap_metric = None NOTE the way I'm setting es_episode_narrative.cluster_memberships below precludes restricting by umap_metric 
    print(f'Begin populate_bertopic_clusters script for show_key={show_key}')

    # load bertopic_data files 
    bertopic_model_list_response = esr.list_bertopic_models(show_key)
    bertopic_model_ids = bertopic_model_list_response['bertopic_model_ids']
    print(f'Found {len(bertopic_model_ids)} bertopic_model_ids matching show_key={show_key}.')

    # initialize dict of narrative-speaker-groups per episode
    epnarr_spkrgrps_to_model_clusters = {}
    simple_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
    if 'episodes' not in simple_episodes_response:
        print(f'Failure to populate_bertopic_clusters for show_key={show_key}: /fetch_simple_episodes returned no episodes')
        return None
    for episode in simple_episodes_response['episodes']:
        e_key = episode['episode_key']
        narrative_sequences_response = esr.fetch_narrative_sequences(ShowKey(show_key), e_key)
        if 'narrative_sequences' not in narrative_sequences_response:
            print(f'Unable to populate_bertopic_clusters for e_key={e_key} show_key={show_key}: /fetch_narrative_sequences returned no narrative_sequences. Skipping episode.')
            continue
        ep_narrs = narrative_sequences_response['narrative_sequences']
        epnarr_spkrgrps_to_model_clusters[e_key] = {narr['speaker_group']:[] for narr in ep_narrs}
    
    # populate episode-narrative-speaker-groups with any model_clusters of which they are a member
    for bertopic_model_id in bertopic_model_ids:
        df = pd.read_csv(f'{BERTOPIC_DATA_DIR}/{show_key}/{bertopic_model_id}.csv', sep='\t')
        # model_id = bertopic_model_id.removesuffix('.csv')
        for _, row in df.iterrows():
            e_key = str(row['episode_key'])
            spkr_grp = row['speaker_group']
            model_cluster = {}
            model_cluster['model_id'] = bertopic_model_id
            model_cluster['model_cluster_id'] = f"{bertopic_model_id}__{e_key}_{row['cluster_id']}__{spkr_grp}"
            model_cluster['probability'] = row['Probability']
            model_cluster['prob_x_wc'] = row['prob_x_wc']
            model_cluster['cluster_title'] = row['cluster_title']
            # convert stringified cluster_keywords back into list
            cluster_keywords = row['cluster_keywords'].split("', '")
            if cluster_keywords:
                cluster_keywords[0] = cluster_keywords[0].removeprefix("['")
                cluster_keywords[len(cluster_keywords)-1] = cluster_keywords[len(cluster_keywords)-1].removesuffix("']")
            model_cluster['cluster_keywords'] = cluster_keywords
            epnarr_spkrgrps_to_model_clusters[e_key][spkr_grp].append(model_cluster)

    # update all cluster_memberships for any given episode-narrative at once, as opposed to piece-meal incrementally (since we don't have any criteria for deleting old mappings)
    attempt_count = 0
    success_count = 0
    failure_count = 0
    for e_key, spkr_grps_to_clusters in epnarr_spkrgrps_to_model_clusters.items():
        for spkr_grp, model_clusters in spkr_grps_to_clusters.items():
            attempt_count += 1
            es_episode_narrative = esqb.fetch_episode_narrative(show_key, e_key, spkr_grp)
            if not es_episode_narrative:
                print(f'Failure to update cluster_memberships for episode narrative: no EsEpisodeNarrativeSequence found matching show_key={show_key} e_key={e_key} spkr_grp={spkr_grp}. Skipping to next episode narrative.')
                failure_count += 1
                continue
            es_episode_narrative.cluster_memberships = model_clusters
            try:
                esqb.save_episode_narrative(es_episode_narrative)
                success_count += 1
            except Exception as e:
                print(f'Failure to update cluster_memberships for episode narrative at show_key={show_key} e_key={e_key} spkr_grp={spkr_grp}: {e}')
                failure_count += 1

    print(f"Finished populating clusters for {len(bertopic_model_ids)} BERTopic models for show_key={show_key}, {success_count} of {attempt_count} mappings populated successfully.")

    report = {"attempt_count": attempt_count, "success_count": success_count, "failure_count": failure_count}
    print(report)
    return report


if __name__ == '__main__':
    main()
