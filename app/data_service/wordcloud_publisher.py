import matplotlib.pyplot as plt
from wordcloud import WordCloud

import app.es.es_read_router as esr
from app.show_metadata import ShowKey


def publish_episode_wordclouds(show_key: str, episode_keys: list, max_words: int = None) -> None:
    print(f'Begin publish_episode_wordcloud for show_key={show_key} len(episode_keys)={len(episode_keys)} max_words={max_words}')
    for episode_key in episode_keys:
        publish_wordcloud(show_key, 'episode', level_key=episode_key, max_words=max_words)


def publish_season_wordclouds(show_key: str, seasons: list, max_words: int = None, include_episodes: bool = False) -> None:
    print(f'Begin publish_season_wordclouds for show_key={show_key} len(seasons)={len(seasons)} max_words={max_words} include_episodes={include_episodes}')
    for season in seasons:
        publish_wordcloud(show_key, 'season', level_key=season, max_words=max_words)
        if include_episodes:
            simple_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key), season=season)
            simple_episodes = simple_episodes_response['episodes']
            publish_episode_wordclouds(show_key, [ep['episode_key'] for ep in simple_episodes], max_words=max_words)


def publish_series_wordcloud(show_key: str, max_words: int = None, include_seasons: bool = False, include_episodes: bool = False) -> None:
    print(f'Begin publish_series_wordcloud for show_key={show_key} max_words={max_words} include_seasons={include_seasons} include_episodes={include_episodes}')
    publish_wordcloud(show_key, 'series', max_words=max_words)
    if include_seasons:
        seasons_response = esr.list_seasons(ShowKey(show_key))
        seasons = seasons_response['seasons']
        publish_season_wordclouds(show_key, [s for s in seasons], max_words=max_words, include_episodes=include_episodes)


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
