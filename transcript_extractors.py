from bs4 import BeautifulSoup
import re

from app.models import Episode, Scene, SceneEvent
import dao
from show_metadata import show_metadata
# from type_system import Episode, Scene, SceneEvent, SceneDialog
from type_system import SceneDialog



CAPTAINS_LOG_PREFIX = "Captain's log"


async def extract_episode_transcript(show_key: str, episode_key: str, transcript_type: str, transcript_soup: BeautifulSoup) -> Episode|Exception:
    print(f'In extract_episode_transcript show_key={show_key} episode_key={episode_key} transcript_type={transcript_type}')
    episode = Episode()
    episode.external_key = episode_key
    # set title to episode_key (temporary hack)
    episode.title = episode_key
    # TODO extract season and sequence_in_season info
    episode.season = 1
    episode.sequence_in_season = 1

    scenes = []
    # scenes_to_events = dict[int, list[SceneEvent]]
    scenes_to_events = {}


    show = None
    try:
        show = await dao.fetch_show(show_key)
    except Exception as e:
         raise Exception(f'Failed to fetch Show having show_key={show_key}: ', e)
    episode.show = show

    # extraction biz logic by show / transcript_type 
    if show_key == 'GoT':
        scene_i = 1

        # Fanon
        if transcript_type == 'Fanon':
            transcript_raw = transcript_soup.find_all('div', class_='mw-parser-output')[0]
            transcript_lines = transcript_raw.find_all('p')

            scene_change_prefixes = ['CUT TO: ', 'EXT. ', 'INT. ']

            first_line = True
            for line in transcript_lines:
                line = line.get_text()

                # preliminary trimming
                line = line.replace(').', '.)')
                line = line.replace('].', '.]')
                # line = line.replace('  ', ' ')
                line = line.strip()

                # first line may not have distinct prefix text but needs to initialize a scene
                if first_line:
                    first_line = False
                    scene = Scene()
                    scene.sequence_in_episode = scene_i
                    scene_events = []
                    scenes_to_events[scene_i] = scene_events
                    scene_i += 1
                    scene.description = line
                    scenes.append(scene)

                # if line starts with a scene_change_prefix -> initialize new scene
                elif line.startswith(tuple(scene_change_prefixes)):
                    prefix_len = 0
                    for pref in scene_change_prefixes:
                        if line.startswith(pref):
                            prefix_len = len(pref)
                            break
                    scene = Scene()
                    scene.sequence_in_episode = scene_i
                    scene_events = []
                    scenes_to_events[scene_i] = scene_events
                    scene_i += 1
                    line_bits = line.split('\n')
                    if len(line_bits) > 1:
                        scene.location = line_bits[0][prefix_len:]
                        scene.description = line_bits[1]
                    else:
                        scene.location = line[prefix_len:]
                    scenes.append(scene)

                # otherwise line is either dialog or context_info within a scene
                else:
                    add_scene_event(line, scene_events)

            # clean up: if first event in scene is nothing but context_info, move it to the scene.description field and delete the event
            # TODO refactor
            # for scene in episode.scenes:
            #     if not scene.description and len(scene.events) > 0 and isinstance(scene.events[0], SceneEvent) and scene.events[0].context_info:
            #         scene.description = scene.events[0].context_info
            #         scene.events = scene.events[1:]

    elif show_key == 'TNG':
        transcript_raw = transcript_soup.find_all('td')[0]
        scene_locations_and_content = transcript_raw.find_all('p')

        scene_i = 1

        # hack to handle initial text that isn't wrapped in p tags (this is f'n annoying)
        first_scene_text_unprocessed = False
        scene = None
        pre_p = transcript_raw.find_all('font')[0]
        locations = pre_p.find_all('b')
        if locations and len(locations) > 0:
            scene = Scene()
            scene.episode = episode
            scene.sequence_in_episode = scene_i
            scene_events = []
            scenes_to_events[scene_i] = scene_events
            scene_i += 1
            location = locations[0].get_text()
            scene.location = location[1:-1]
            print(f'1 adding scene={scene} to scenes for episode={episode}')
            scenes.append(scene)
        else:
            first_scene_text = basic_trim_tng(pre_p.get_text())
            if first_scene_text and len(first_scene_text) > 0:
                first_scene_text_unprocessed = True

        # iterate thru scene locations and lines separated by p tags
        for slac in scene_locations_and_content:
            locations = slac.find_all('b')
            if locations and len(locations) > 0:
                scene = Scene()
                scene.episode = episode
                scene.sequence_in_episode = scene_i
                scene_events = []
                scenes_to_events[scene_i] = scene_events
                scene_i += 1
                location = locations[0].get_text()
                # print(f'location={location}')
                scene.location = location[1:-1]
                print(f'2 adding scene={scene} to scenes for episode={episode}')
                scenes.append(scene)
                if first_scene_text_unprocessed:
                    add_scene_event(first_scene_text, scene_events)
                    first_scene_text_unprocessed = False

            else:
                lines = [line for line in slac.strings]
                for line in lines:
                    # print(f'line={line}')
                    line = line.get_text()
                    line = basic_trim_tng(line)
                    if line and len(line) > 0:
                        if scene:
                            add_scene_event(line, scene_events)
                        else:
                            # another hack to handle initial text that IS wrapped in p tags but precedes scene creation
                            first_scene_text_unprocessed = True
                            first_scene_text = line

    # print(f'After assembling episode={episode}:')
    # for scene in scenes:
    #     print(f'>> scene={scene} of type={scene}')

    return episode, scenes, scenes_to_events


'''
Parse line text to extract sceneDialog and sceneEvent info to be added to scene
'''
def add_scene_event(line: str, scene_events: list[SceneEvent]):
    event = SceneEvent()
    line_bits = line.split(': ', 1)
    # if line contains a colon, we assume it splits a character name and their dialog
    if len(line_bits) > 1:
        # if line contains substrings wrapped in brackets or parens, extract these into context_info
        extract_context_from_dialog(line_bits[0], event, 'spoken_by')
        extract_context_from_dialog(line_bits[1], event, 'line')
        scene_events.append(event)
    # otherwise assume the line is context_info between lines of dialog within a scene
    else:
        if line and line[0] in ['[', '(']:
            line = line[1:-1]
        event.context_info = line
        scene_events.append(event)


# '''
# Parse line text to extract sceneDialog and sceneEvent info to be added to scene
# '''
# def add_scene_event(line: str, scene: Scene):
#     event = SceneEvent()
#     line_bits = line.split(': ', 1)
#     # if line contains a colon, we assume it splits a character name and their dialog
#     if len(line_bits) > 1:
#         # if line contains substrings wrapped in brackets or parens, extract these into context_info
#         extract_context_from_dialog(line_bits[0], event, 'spoken_by')
#         extract_context_from_dialog(line_bits[1], event, 'line')
#         scene.events.append(event)
#     # otherwise assume the line is context_info between lines of dialog within a scene
#     else:
#         if line and line[0] in ['[', '(']:
#             line = line[1:-1]
#         event.context_info = line
#         scene.events.append(event)


'''
If line contains substrings wrapped in brackets or parens, extract these into context_info
'''
def extract_context_from_dialog(raw_text: str, event: SceneEvent, field: str) -> None:
    ctx_info = re.findall(r'[\(\[].*?[\)\]]', raw_text)
    if len(ctx_info) == 0:
        processed_text = raw_text
    else:
        processed_text = re.sub("[\(\[].*?[\)\]]", "", raw_text)
        processed_text = processed_text.strip()
        processed_text = processed_text.replace('  ', ' ')
        if not event.context_info:
            event.context_info = ''
        for ci in ctx_info:
            event.context_info = f'{event.context_info}{ci[1:-1]} '
        event.context_info = event.context_info.strip()

    if field == 'spoken_by':
        event.dialogue_spoken_by = processed_text
    else:
        event.dialogue_text = processed_text


def basic_trim_tng(line: str) -> str:
    line = line.replace('\n', ' ')
    line = line.replace('\r', ' ')
    line = line.replace('  ', ' ')
    line = line.strip()
    if line.startswith(CAPTAINS_LOG_PREFIX):
        line = f'PICARD: {line}'
    return line



# class TranscriptExtractor(object):
#     def __init__(self, show_key: str, transcript_type: str, transcript_url: str):
#         self.show_key = show_key
#         self.transcript_type = transcript_type
#         self.transcript_url = transcript_url

#     def extract_transcript(self, transcript_soup: BeautifulSoup):
#         episode = Episode()

#         # extraction biz logic by show / transcript_type 
#         # GoT
#         if self.show_key == 'GoT':
#             # Fanon
#             if self.transcript_type == 'Fanon':
#                 episode.show = self.show_key
#                 transcript_raw = transcript_soup.find_all('div', class_='mw-parser-output')[0]
#                 transcript_lines = transcript_raw.find_all('p')

#                 scene_change_prefixes = ['CUT TO: ', 'EXT. ', 'INT. ']

#                 first_line = True
#                 for line in transcript_lines:
#                     line = line.get_text()

#                     # preliminary trimming
#                     line = line.replace(').', '.)')
#                     line = line.replace('].', '.]')
#                     # line = line.replace('  ', ' ')
#                     line = line.strip()

#                     # first line may not have distinct prefix text but needs to initialize a scene
#                     if first_line:
#                         first_line = False
#                         scene = Scene()
#                         scene.description = line
#                         episode.scenes.append(scene)

#                     # if line starts with a scene_change_prefix -> initialize new scene
#                     elif line.startswith(tuple(scene_change_prefixes)):
#                         prefix_len = 0
#                         for pref in scene_change_prefixes:
#                             if line.startswith(pref):
#                                 prefix_len = len(pref)
#                                 break
#                         scene = Scene()
#                         line_bits = line.split('\n')
#                         if len(line_bits) > 1:
#                             scene.location = line_bits[0][prefix_len:]
#                             scene.description = line_bits[1]
#                         else:
#                             scene.location = line[prefix_len:]
#                         episode.scenes.append(scene)

#                     # otherwise line is either dialog or context_info within a scene
#                     else:
#                         add_dialog_or_event(line, scene)

#                 # clean up: if first event in scene is nothing but context_info, move it to the scene.description field and delete the event
#                 for scene in episode.scenes:
#                     if not scene.description and len(scene.events) > 0 and isinstance(scene.events[0], SceneEvent) and scene.events[0].context_info:
#                         scene.description = scene.events[0].context_info
#                         scene.events = scene.events[1:]

#             # extract title from url (temporary hack)
#             episode.transcript_url = self.transcript_url
#             title = self.transcript_url.removeprefix(show_metadata[self.show_key]['show_transcripts_domain'])
#             title = title.removeprefix(show_metadata[self.show_key]['episode_subdir'])
#             transcript_type_string_match = show_metadata[self.show_key]['transcript_types'][self.transcript_type]
#             title = title.replace(transcript_type_string_match, '')
#             episode.title = title

#         elif self.show_key == 'TNG':
#             episode.show = self.show_key
#             transcript_file = self.transcript_url.split('/')[-1]
#             episode.title = transcript_file.split('.')[0]

#             transcript_raw = transcript_soup.find_all('td')[0]
#             scene_locations_and_content = transcript_raw.find_all('p')

#             # hack to handle initial text that isn't wrapped in p tags (this is f'n annoying)
#             first_scene_text_unprocessed = False
#             scene = None
#             pre_p = transcript_raw.find_all('font')[0]
#             locations = pre_p.find_all('b')
#             if locations and len(locations) > 0:
#                 scene = Scene()
#                 location = locations[0].get_text()
#                 scene.location = location[1:-1]
#                 episode.scenes.append(scene)
#             else:
#                 first_scene_text = basic_trim_tng(pre_p.get_text())
#                 if first_scene_text and len(first_scene_text) > 0:
#                     first_scene_text_unprocessed = True

#             # iterate thru scene locations and lines separated by p tags
#             for slac in scene_locations_and_content:
#                 locations = slac.find_all('b')
#                 if locations and len(locations) > 0:
#                     scene = Scene()
#                     location = locations[0].get_text()
#                     # print(f'location={location}')
#                     scene.location = location[1:-1]
#                     episode.scenes.append(scene)
#                     if first_scene_text_unprocessed:
#                         add_dialog_or_event(first_scene_text, scene)
#                         first_scene_text_unprocessed = False

#                 else:
#                     lines = [line for line in slac.strings]
#                     for line in lines:
#                         # print(f'line={line}')
#                         line = line.get_text()
#                         line = basic_trim_tng(line)
#                         if line and len(line) > 0:
#                             if scene:
#                                 add_dialog_or_event(line, scene)
#                             else:
#                                 # another hack to handle initial text that IS wrapped in p tags but precedes scene creation
#                                 first_scene_text_unprocessed = True
#                                 first_scene_text = line

#         print('----------------------------------------------------------------------------')
#         print(f'len(episode.scenes)={len(episode.scenes)}')

#         episode.external_id = episode.title # TODO hmmm

#         return episode
    

# '''
# Parse line text to extract sceneDialog and sceneEvent info to be added to scene
# '''
# def add_dialog_or_event(line: str, scene: Scene):
#     line_bits = line.split(': ', 1)
#     # if line contains a colon, we assume it splits a character name and their dialog
#     if len(line_bits) > 1:
#         dialog = SceneDialog()
#         # if line contains substrings wrapped in brackets or parens, extract these into context_info
#         extract_context_info_from_dialog(line_bits[0], dialog, 'spoken_by')
#         extract_context_info_from_dialog(line_bits[1], dialog, 'line')
#         scene.events.append(dialog)
#     # otherwise assume the line is context_info between lines of dialog within a scene
#     else:
#         event = SceneEvent()
#         if line and line[0] in ['[', '(']:
#             line = line[1:-1]
#         event.context_info = line
#         scene.events.append(event)


# '''
# If line contains substrings wrapped in brackets or parens, extract these into context_info
# '''
# def extract_context_info_from_dialog(raw_text: str, dialog: SceneDialog, field: str) -> None:
#     ctx_info = re.findall(r'[\(\[].*?[\)\]]', raw_text)
#     if len(ctx_info) == 0:
#         processed_text = raw_text
#     else:
#         processed_text = re.sub("[\(\[].*?[\)\]]", "", raw_text)
#         processed_text = processed_text.strip()
#         processed_text = processed_text.replace('  ', ' ')
#         if not dialog.context_info:
#             dialog.context_info = ''
#         for ci in ctx_info:
#             dialog.context_info = f'{dialog.context_info}{ci[1:-1]} '
#         dialog.context_info = dialog.context_info.strip()

#     if field == 'spoken_by':
#         dialog.spoken_by = processed_text
#     else:
#         dialog.line = processed_text
