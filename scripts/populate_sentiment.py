import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import app.data_service.sentiment_populator as sp
import app.routers.es_read_router as esr
from app.show_metadata import ShowKey


def main():
    # parse script params
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--analyzer", "-a", help="Analyzer", required=True)
    parser.add_argument("--episode_keys", "-e", help="Episode keys", required=False)
    parser.add_argument("--season", "-n", help="Season", required=False)
    parser.add_argument("--scene_level", "-c", help="Scene level", required=False)
    # parser.add_argument("--line_level", "-l", help="Line level", required=False)
    parser.add_argument("--overwrite_csv", "-o", help="Overwrite CSV file", required=False)
    parser.add_argument("--write_to_es", "-w", help="Write to es", required=False)
    args = parser.parse_args()

    # TODO haven't solved for setting this correctly, requires altering exit_if_unauthorized to run 
    user_dependency = None

    season = None
    scene_level = False
    line_level = False
    overwrite_csv = False
    write_to_es = False
    if args.season: 
        season = args.season
    if args.scene_level: 
        scene_level = args.scene_level
    # if args.line_level: 
    #     line_level = args.line_level
    if args.overwrite_csv: 
        overwrite_csv = args.overwrite_csv
    if args.write_to_es: 
        write_to_es = args.write_to_es

    if args.episode_keys:
        e_keys = args.episode_keys.split(',')
    elif args.season:
        simple_episodes_response = esr.fetch_simple_episodes(ShowKey(args.show_key), user_dependency, season=season)
        e_keys = [se['episode_key'] for se in simple_episodes_response['episodes']]
    else:
        print(f'Either `episode_keys` (-e) or `season` (-n) is required, populating sentiment for an entire series in a single job is currently not supported')
        return 

    for e_key in e_keys:
        sp.populate_episode_sentiment(args.show_key, e_key, args.analyzer, scene_level=scene_level, overwrite_csv=overwrite_csv, write_to_es=write_to_es)


if __name__ == '__main__':
    main()
