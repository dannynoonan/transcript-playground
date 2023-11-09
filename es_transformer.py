
from app.models import Episode
from es_model import EsEpisodeTranscript, EsScene, EsSceneEvent


async def to_es_episode(episode: Episode) -> EsEpisodeTranscript:
    es_episode = EsEpisodeTranscript(
        show_key=episode.show_key, episode_key=episode.external_key, title=episode.title, season=episode.season, 
        sequence_in_season=episode.sequence_in_season, air_date=episode.air_date)
    es_episode.meta.id = f'{episode.show_key}_{episode.external_key}'
    es_episode.scenes = []
    for scene in episode.scenes:
        es_scene = EsScene(location=scene.location, description=scene.description)
        es_scene.scene_events = []
        for scene_event in scene.events:
            es_scene_event = EsSceneEvent(
                context_info=scene_event.context_info, spoken_by=scene_event.dialogue_spoken_by, dialog=scene_event.dialogue_text)
            es_scene.scene_events.append(es_scene_event)
        es_episode.scenes.append(es_scene)

    return es_episode
