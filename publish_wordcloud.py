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
    parser.add_argument("--include_children", "-c", help="Include children", required=False)
    parser.add_argument("--max_words", "-m", help="Max words", required=False)
    args = parser.parse_args()
    show_key = args.show_key
    episode_keys = None
    season = None
    include_children = None
    if args.episode_keys:
        episode_keys = args.episode_keys
    if args.season:
        season = args.season
    if args.max_words:
        max_words = args.max_words
    else:
        max_words = 50
    if args.include_children:
        include_children = args.include_children

    if episode_keys:
        e_keys = episode_keys.split(',')
        for e_key in e_keys:
            publish_wordcloud(show_key, 'episode', level_key=e_key, max_words=max_words)
    elif season:
        publish_wordcloud(show_key, 'season', level_key=season, max_words=max_words)
        if include_children:
            simple_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key), season=season)
            simple_episodes = simple_episodes_response['episodes']
            for ep in simple_episodes:
                publish_wordcloud(show_key, 'episode', level_key=ep['episode_key'], max_words=max_words)
    else:
        publish_wordcloud(show_key, 'series', max_words=max_words)
        # TODO support series-wide wordcloud?
        if include_children:
            print('`include_children` only valid at season level, not for entire series')


# def publish_episode_wordcloud(show_key: str, episode_key: str, max_words: int = None) -> None:
#     episode_keywords_response = esr.keywords_by_episode(ShowKey(show_key), episode_key, exclude_speakers=True)
#     keywords = episode_keywords_response['keywords']
#     kws_at_strength = []
#     for kw in keywords:
#         kw_vector = [kw['term']] * round(kw['score'])
#         kws_at_strength.extend(kw_vector)
#     kws_at_strength

#     wordcloud = WordCloud(background_color='white', width=512, height=384, max_words=max_words, collocations=False).generate(' '.join(kws_at_strength))
#     plt.imshow(wordcloud) # image show
#     plt.axis('off') # to off the axis of x and y

#     img_path = f'static/wordclouds/{show_key}/{show_key}_{episode_key}.png'
#     plt.savefig(img_path)
#     # plt.show()


def publish_wordcloud(show_key: str, level: str, level_key: str = None, max_words: int = None) -> None:
    print(f'Begin publish_wordcloud for show_key={show_key} level={level} level_key={level_key}, max_words={max_words}')

    if level in ['episode', 'season'] and not level_key:
        print(f'Failure to publish_wordcloud, level_key is required if level={level}')
        return
    
    file_name = show_key

    if level == 'episode':
        episode_keywords_response = esr.keywords_by_episode(ShowKey(show_key), level_key, exclude_speakers=True)
        keywords = episode_keywords_response['keywords']
        file_name = f'{show_key}_{level_key}'
        multiplier = 'score'
    elif level == 'season':
        season_keywords_response = esr.keywords_by_corpus(ShowKey(show_key), season=level_key, exclude_speakers=True)
        keywords = season_keywords_response['keywords']
        file_name = f'{show_key}_SEASON{level_key}'
        multiplier = 'ttf'
    elif level == 'series':
        series_keywords_response = esr.keywords_by_corpus(ShowKey(show_key), exclude_speakers=True) # level_key is ignored
        keywords = series_keywords_response['keywords']
        file_name = f'{show_key}_SERIES'
        multiplier = 'ttf'

    kws_at_strength = []
    for kw in keywords:
        kw_vector = [kw['term']] * round(kw[multiplier])
        kws_at_strength.extend(kw_vector)
    kws_at_strength

    wordcloud = WordCloud(background_color='white', width=512, height=384, max_words=max_words, collocations=False).generate(' '.join(kws_at_strength))
    plt.imshow(wordcloud) # image show
    plt.axis('off') # to off the axis of x and y

    img_path = f'static/wordclouds/{show_key}/{file_name}.png'
    plt.savefig(img_path)


if __name__ == '__main__':
    main()
