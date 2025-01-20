from dash import callback, Input, Output
import pandas as pd

from app.auth import ADMIN_USER
import app.fig_builder.plotly_bar as pb
import app.routers.es_read_router as esr
from app.show_metadata import ShowKey


############ character summary callback
@callback(
    Output('character-summary', 'children'),
    Output('character-alt-names', 'children'),
    Output('character-actor-names', 'children'),
    Output('character-season-count', 'children'),
    Output('character-episode-count', 'children'),
    Output('character-scene-count', 'children'),
    Output('character-line-count', 'children'),
    Output('character-word-count', 'children'),
    Output('character-top-mbti', 'children'),
    Output('character-top-dnda', 'children'),
    Input('show-key', 'data'),
    Input('speaker-key', 'value')
)    
def render_character_summary(show_key: str, speaker_key: str):
    print(f'in render_character_summary, show_key={show_key} speaker_key={speaker_key}')

    # core speaker data
    speaker_es_response = esr.fetch_speaker(ShowKey(show_key), speaker_key, ADMIN_USER, include_seasons=True, include_episodes=True)
    if 'speaker' not in speaker_es_response:
        err_msg = f'no episode matching show_key={show_key} speaker_key={speaker_key}'
        print(err_msg)
        return None, None, None, None, None, None, None, None  
    # summary = speaker_key
    speaker = speaker_es_response['speaker']
    alt_names = None
    if 'alt_names' in speaker:
        alt_names = [name for name in speaker['alt_names'] if name.upper() != speaker_key]
        alt_names = ', '.join(alt_names)
    actor_names = None
    if 'actor_names' in speaker:
        actor_names = speaker['actor_names']
        actor_names = ', '.join(actor_names)
    season_count = None
    if 'season_count' in speaker:
        season_count = speaker['season_count']
    episode_count = None
    if 'episode_count' in speaker:
        episode_count = speaker['episode_count']
    scene_count = None
    if 'scene_count' in speaker:
        scene_count = speaker['scene_count']
    line_count = None
    if 'line_count' in speaker:
        line_count = speaker['line_count']
    word_count = None
    if 'word_count' in speaker:
        word_count = speaker['word_count']
    top_mbti = None
    if 'topics_mbti' in speaker:
        top_mbti = [t['topic_key'] for t in speaker['topics_mbti']]
        top_mbti = top_mbti[:3]
        top_mbti = ', '.join(top_mbti)
    top_dnda = None
    if 'topics_dnda' in speaker:
        top_dnda = [t['topic_key'] for t in speaker['topics_dnda']]
        top_dnda = top_dnda[:2]
        top_dnda = ', '.join(top_dnda)

    return speaker_key, alt_names, actor_names, season_count, episode_count, scene_count, line_count, word_count, top_mbti, top_dnda


############ character espisode callback
@callback(
    Output('character-series-hist-bar', 'figure'),
    # Output('character-series-hist-dt', 'children'),
    Input('show-key', 'data'),
    Input('speaker-key', 'value'),
    Input('granularity', 'value')
)   
def render_character_series_histogram(show_key: str, speaker_key: str, granularity: str):
    print(f'in render_character_series_histogram, show_key={show_key} speaker_key={speaker_key} granularity={granularity}')

    response = esr.generate_speaker_episode_bar_sequence(ShowKey(show_key), speaker_key, ADMIN_USER)
    if 'speaker_episode_bar_sequence' not in response:
        # TODO
        return None
    
    df = pd.DataFrame(response['speaker_episode_bar_sequence'])
    df.sort_values(['season', 'sequence_in_season'], ascending=[True, True], inplace=True)
    # print(f'df={df}')

    # build speaker episode histogram chart
    character_series_hist_bar = pb.build_character_series_hist(show_key, speaker_key, df, granularity)

    return character_series_hist_bar
