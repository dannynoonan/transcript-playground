from dash import callback, Input, Output
import os
import pandas as pd

from app.app_metadata import SENTIMENT_DATA_DIR
from app.auth import ADMIN_USER
import app.fig_builder.plotly_bar as pb
import app.routers.es_read_router as esr
# import app.page_builder_service.character_page_service as cps
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


############ character series hist-bar chart callback
@callback(
    Output('character-series-hist-bar', 'figure'),
    # Output('character-series-hist-dt', 'children'),
    Input('show-key', 'data'),
    Input('speaker-key', 'value'),
    Input('granularity', 'value'),
    Input('focus', 'value')
)   
def render_character_series_histogram(show_key: str, speaker_key: str, granularity: str, focus: str):
    print(f'in render_character_series_histogram, show_key={show_key} speaker_key={speaker_key} granularity={granularity} focus={focus}')

    response = esr.generate_speaker_episode_bar_sequence(ShowKey(show_key), speaker_key, ADMIN_USER)
    if 'speaker_episode_bar_sequence' not in response:
        # TODO
        return None
    
    # load into df, sort / rank / tweak cell contents
    speaker_df = pd.DataFrame(response['speaker_episode_bar_sequence'])
    speaker_df['air_date'] = speaker_df['air_date'].apply(lambda x: x[:10])
    # df.rename(columns={'top_locations': 'locations', 'top_companions': 'companions'}, inplace=True)
    # print(f'df.columns={df.columns}')
    speaker_df['scene_count_rank'] = speaker_df['scene_count'].rank(ascending=False, method='min').astype(int)
    speaker_df['line_count_rank'] = speaker_df['line_count'].rank(ascending=False, method='min').astype(int)
    speaker_df['word_count_rank'] = speaker_df['word_count'].rank(ascending=False, method='min').astype(int)
    speaker_df.sort_values(['season', 'sequence_in_season'], ascending=[True, True], inplace=True)
    speaker_df['sequence'] = range(1, len(speaker_df) + 1)

    # build speaker episode histogram chart
    character_series_hist_bar = pb.build_speaker_series_hist(show_key, speaker_key, speaker_df, granularity, color_col=focus)

    return character_series_hist_bar


############ character series sentiment bar chart callback
@callback(
    Output('character-series-sentiment-bar', 'figure'),
    # Output('character-series-sentiment-dt', 'children'),
    Input('show-key', 'data'),
    Input('speaker-key', 'value'),
    Input('emotion', 'value')
    # Input('focus', 'value')
)   
def render_character_series_sentiment_hist(show_key: str, speaker_key: str, emotion: str):
    print(f'in render_character_series_sentiment_hist, show_key={show_key} speaker_key={speaker_key} emotion={emotion}')

    response = esr.generate_speaker_episode_bar_sequence(ShowKey(show_key), speaker_key, ADMIN_USER)
    if 'speaker_episode_bar_sequence' not in response:
        # TODO
        return None
    
    # load into df, sort / rank / tweak cell contents
    speaker_df = pd.DataFrame(response['speaker_episode_bar_sequence'])

    speaker_sent_file_path = f'{SENTIMENT_DATA_DIR}/{show_key}/speakers/openai_emo/{show_key}_{speaker_key}.csv'    
    if not os.path.isfile(speaker_sent_file_path):
        # TODO
        print(f'Failure to render_character_series_sentiment_hist, no file found at speaker_sent_file_path={speaker_sent_file_path}')
        return None

    # merge episode-level speaker emotion averages into full episode listing df
    speaker_sent_df = pd.read_csv(speaker_sent_file_path, sep=',')

    # keep either (a) the highest scoring emotions per episode or (b) the specific emotion specified
    if emotion == 'Highest':
        # limit df to the highest scoring emotion per episode
        speaker_emo_df = speaker_sent_df[speaker_sent_df['high_score'] == True]
    else:
        # limit df to a single selected emotion across episodes
        speaker_emo_df = speaker_sent_df[speaker_sent_df['emotion'] == emotion]
        
    speaker_emo_df = speaker_emo_df[['emotion', 'score', 'episode_key']]
    speaker_emo_df['episode_key'] = speaker_emo_df['episode_key'].astype(str) # TODO grumble grumble why why
    # print(f'speaker_key={speaker_key} emotion={emotion} len(speaker_emo_df={len(speaker_emo_df)})')
    speaker_merged_df = pd.merge(speaker_df, speaker_emo_df, on='episode_key', how='outer')
    # print(f'len(speaker_merged_df={len(speaker_merged_df)})')
    # print(f'speaker_merged_df={speaker_merged_df})')

    speaker_merged_df['air_date'] = speaker_merged_df['air_date'].apply(lambda x: x[:10])
    # df.rename(columns={'top_locations': 'locations', 'top_companions': 'companions'}, inplace=True)
    # print(f'df.columns={speaker_merged_df.columns}')
    speaker_merged_df['scene_count_rank'] = speaker_merged_df['scene_count'].rank(ascending=False, method='min').astype(int)
    speaker_merged_df['line_count_rank'] = speaker_merged_df['line_count'].rank(ascending=False, method='min').astype(int)
    speaker_merged_df['word_count_rank'] = speaker_merged_df['word_count'].rank(ascending=False, method='min').astype(int)
    speaker_merged_df.sort_values(['season', 'sequence_in_season'], ascending=[True, True], inplace=True)
    speaker_merged_df['sequence'] = range(1, len(speaker_merged_df) + 1)

    # build speaker episode histogram chart
    character_series_sentiment_bar = pb.build_speaker_series_sentiment_hist(show_key, speaker_key, speaker_merged_df, emotion)

    return character_series_sentiment_bar
