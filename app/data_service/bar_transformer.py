from app.auth import ADMIN_USER
import app.routers.es_read_router as esr
from app.show_metadata import ShowKey


def generate_speaker_episode_bar_sequence(show_key: ShowKey, speaker_name: str) -> list:
    '''
    TODO
    '''
    speaker_episode_bar_sequence = []

    # load all episodes into dict
    response = esr.fetch_simple_episodes(show_key, ADMIN_USER)
    if 'episodes' not in response:
        print(f'Failure to fetch_simple_episodes for show_key={show_key.value}')
        return speaker_episode_bar_sequence
    all_episodes = {e['episode_key']:e for e in response['episodes']}

    # load speaker episode data into list of dicts
    response = esr.fetch_speaker(show_key, speaker_name, ADMIN_USER, include_episodes=True)
    if 'speaker' not in response or 'episodes' not in response['speaker']:
        print(f'Failure to fetch episodes for speaker={speaker_name}')
        return speaker_episode_bar_sequence
    for e in response['speaker']['episodes']:
        speaker_episode_row = dict(episode_key=e['episode_key'], season=e['season'], sequence_in_season=e['sequence_in_season'], 
                                   title=e['title'], air_date=e['air_date'], mbti=e['topics_mbti'][0]['topic_key'], dnda=e['topics_dnda'][0]['topic_key'], 
                                   scene_count=e['scene_count'], line_count=e['line_count'], word_count=e['word_count'], 
                                   locations=e['top_locations'], top_location=e['top_locations'][0], 
                                   companions=e['top_companions'], top_companion=e['top_companions'][0], 
                                #    similar_speakers=e['similar_speakers'][0],
                                   openai_word_count=e['openai_word_count'], agg_score=e['agg_score'])
        speaker_episode_bar_sequence.append(speaker_episode_row)
        # delete episode from all_episodes after loading
        del all_episodes[e['episode_key']]

    # episodes remaining in all_episodes are the ones speaker did not speak in, 
    # add corresponding speaker_episode_row to speaker_episode_bar_sequence for each.
    for e_key, e in all_episodes.items():
        speaker_episode_row = dict(episode_key=e_key, season=e['season'], sequence_in_season=e['sequence_in_season'], 
                                   title=e['title'], air_date=e['air_date'], 
                                #    topics_mbti=e['topics_mbti'], topics_dnda=e['topics_dnda'], 
                                   scene_count=0, line_count=0, word_count=0, openai_word_count=0, agg_score=0)
        speaker_episode_bar_sequence.append(speaker_episode_row)

    return speaker_episode_bar_sequence
