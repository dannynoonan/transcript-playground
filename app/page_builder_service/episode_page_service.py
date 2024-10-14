import app.es.es_read_router as esr
from app.show_metadata import ShowKey


def generate_episode_dropdown_options(show_key: str) -> list:
    all_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
    all_episodes = all_episodes_response['episodes']
    episode_dropdown_options = []
    for episode in all_episodes:
        label = f"{episode['title']} (S{episode['season']}:E{episode['sequence_in_season']})"
        episode_dropdown_options.append({'label': label, 'value': episode['episode_key']})
    
    return episode_dropdown_options
