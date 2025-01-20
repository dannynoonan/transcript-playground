from app.auth import ADMIN_USER
import app.routers.es_read_router as esr
from app.show_metadata import ShowKey


def generate_speaker_episode_bar_sequence(show_key: ShowKey, speaker_name: str) -> list:
    '''
    TODO
    '''
    response = esr.fetch_speaker(show_key, speaker_name, ADMIN_USER, include_episodes=True)

    speaker_episode_bar_sequence = []
    if 'speaker' not in response or 'episodes' not in response['speaker']:
        print(f'Failure to fetch episodes for speaker={speaker_name}')
        return speaker_episode_bar_sequence
    for episode in response['speaker']['episodes']:
        speaker_episode_row = dict(episode_key=episode['episode_key'], season=episode['season'], sequence_in_season=episode['sequence_in_season'], 
                                   title=episode['title'], air_date=episode['air_date'], 
                                #    topics_mbti=episode['topics_mbti'], topics_dnda=episode['topics_dnda'], 
                                   scene_count=episode['scene_count'], line_count=episode['line_count'], word_count=episode['word_count'], 
                                   openai_word_count=episode['openai_word_count'], agg_score=episode['agg_score'])
        speaker_episode_bar_sequence.append(speaker_episode_row)

    return speaker_episode_bar_sequence
