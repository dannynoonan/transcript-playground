
from app.models import Episode, Scene, SceneEvent
from es_model import EsEpisodeTranscript, EsScene, EsSceneEvent


async def to_es_episode(episode: Episode) -> EsEpisodeTranscript:
    es_episode = EsEpisodeTranscript(
        show_key=episode.show_key, episode_key=episode.external_key, title=episode.title, season=episode.season, 
        sequence_in_season=episode.sequence_in_season, air_date=episode.air_date)
    flattened_text = f'{episode.title}\n\n'
    es_episode.meta.id = f'{episode.show_key}_{episode.external_key}'
    es_episode.scenes = []
    for scene in episode.scenes:
        # es_episode.add_scene(scene.location, scene.description)
        es_scene = EsScene(location=scene.location, description=scene.description)
        flattened_scene = to_flattened_scene(scene)
        # if scene.location:
        #     flattened_scene += f'{scene.location}\n'
        # if scene.description:
        #     flattened_scene += f'[{scene.description}]\n'
        # flattened_scene += '\n'
        es_scene.scene_events = []
        for scene_event in scene.events:
            es_scene_event = EsSceneEvent(
                context_info=scene_event.context_info, spoken_by=scene_event.dialogue_spoken_by, dialog=scene_event.dialogue_text)
            es_scene.scene_events.append(es_scene_event)
            flattened_scene += to_flattened_scene_event(scene_event)
            # if scene_event.context_info:
            #     flattened_scene += f'[{scene_event.context_info}]\n'
            # if scene_event.dialogue_spoken_by:
            #     flattened_scene += f'{scene_event.dialogue_spoken_by}: {scene_event.dialogue_text}\n'
        es_episode.scenes.append(es_scene)
        flattened_text += flattened_scene + '\n\n'

    es_episode.flattened_text = flattened_text
    return es_episode


def to_flattened_scene(scene: Scene) -> str:
    flattened_scene = ''
    if scene.location:
        flattened_scene += f'{scene.location}\n'
    if scene.description:
        flattened_scene += f'[{scene.description}]\n'
    flattened_scene += '\n'
    return flattened_scene


def to_flattened_scene_event(scene_event: SceneEvent) -> str:
    flattened_scene_event = ''
    if scene_event.context_info:
        flattened_scene_event += f'[{scene_event.context_info}]\n'
    if scene_event.dialogue_spoken_by:
        flattened_scene_event += f'{scene_event.dialogue_spoken_by}: {scene_event.dialogue_text}\n'
    return flattened_scene_event
