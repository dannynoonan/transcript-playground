import argparse
import matplotlib.pyplot as plt
# import os
# import pandas as pd
# import plotly.graph_objs as go
from wordcloud import WordCloud

import app.es.es_read_router as esr
from app.show_metadata import ShowKey


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_key", "-s", help="Show key", required=True)
    parser.add_argument("--episode_keys", "-e", help="Episode keys", required=False)
    parser.add_argument("--season", "-n", help="Season", required=False)
    parser.add_argument("--max_words", "-m", help="Max words", required=False)
    args = parser.parse_args()
    show_key = args.show_key
    episode_keys = None
    season = None
    if args.episode_keys:
        episode_keys = args.episode_keys
    if args.season:
        season = args.season
    if args.max_words:
        max_words = args.max_words
    else:
        max_words = 50

    if episode_keys:
        e_keys = episode_keys.split(',')
        for e_key in e_keys:
            publish_episode_wordcloud(show_key, e_key, max_words=max_words)
    elif season:
        simple_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key), season=season)
        simple_episodes = simple_episodes_response['episodes']
        for ep in simple_episodes:
            publish_episode_wordcloud(show_key, ep['episode_key'], max_words=max_words)
    else:
        print(f'Either `episode_keys` (-e) or `season` (-n) is required, populating wordclouds for an entire series in a single job is currently not supported')
        return 

    
def publish_episode_wordcloud(show_key: str, episode_key: str, max_words: int = None) -> None:
    episode_keywords_response = esr.keywords_by_episode(ShowKey(show_key), episode_key, exclude_speakers=True)
    keywords = episode_keywords_response['keywords']
    kws_at_strength = []
    for kw in keywords:
        kw_vector = [kw['term']] * round(kw['score'])
        kws_at_strength.extend(kw_vector)
    kws_at_strength

    wordcloud = WordCloud(background_color='white', width=512, height=384, max_words=max_words, collocations=False).generate(' '.join(kws_at_strength))
    plt.imshow(wordcloud) # image show
    plt.axis('off') # to off the axis of x and y

    img_path = f'static/wordclouds/{show_key}/{show_key}_{episode_key}.png'
    plt.savefig(img_path)
    # plt.show()


if __name__ == '__main__':
    main()
