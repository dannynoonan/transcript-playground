import argparse

import app.data_service.wordcloud_publisher as wp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--episode_keys", "-e", help="Episode keys", required=False)
    parser.add_argument("--seasons", "-n", help="Seasons", required=False)
    parser.add_argument("--include_children", "-c", help="Include children", required=False)
    parser.add_argument("--max_words", "-m", help="Max words", required=False)
    args = parser.parse_args()

    show_key = args.show_key
    episode_keys = None
    seasons = None
    include_children = None
    if args.episode_keys:
        episode_keys = args.episode_keys
    if args.seasons:
        seasons = args.seasons
    if args.max_words:
        max_words = args.max_words
    else:
        max_words = 50
    if args.include_children:
        include_children = args.include_children


    if episode_keys:
        e_keys = episode_keys.split(',')
        wp.publish_episode_wordclouds(show_key, e_keys, max_words=max_words)
    elif seasons:
        wp.publish_season_wordclouds(show_key, seasons, max_words=max_words, include_episodes=include_children)
    else:
        wp.publish_series_wordcloud(show_key, max_words=max_words, include_seasons=include_children, include_episodes=include_children)


if __name__ == '__main__':
    main()
