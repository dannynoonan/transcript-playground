from bs4 import BeautifulSoup
import re

from type_system import Episode, Scene, SceneEvent, SceneDialog


class TranscriptExtractor(object):
    def __init__(self, show_key: str, transcript_type: str):
        self.show_key = show_key
        self.transcript_type = transcript_type

    def extract_transcript(self, transcript_soup: BeautifulSoup):
        episode = Episode()

        # extraction biz logic specific to the GoT-Fanon transcript type
        if self.transcript_type == 'Fanon':
            episode.show = self.show_key
            transcript_raw = transcript_soup.find_all('div', class_='mw-parser-output')[0]
            transcript_lines = transcript_raw.find_all('p')

            scene_change_prefixes = ['CUT TO: ', 'EXT. ', 'INT. ']

            first_line = True
            for line in transcript_lines:
                line = line.get_text()

                # preliminary trimming
                line = line.replace(").", ".)")
                line = line.replace("].", ".]")
                line = line.strip()

                # first line may not have distinct prefix text but needs to initialize a scene
                if first_line:
                    first_line = False
                    scene = Scene()
                    scene.description = line
                    episode.scenes.append(scene)

                # if line starts with a scene_change_prefix -> initialize new scene
                elif line.startswith(tuple(scene_change_prefixes)):
                    prefix_len = 0
                    for pref in scene_change_prefixes:
                        if line.startswith(pref):
                            prefix_len = len(pref)
                            break
                    scene = Scene()
                    line_bits = line.split('\n')
                    if len(line_bits) > 1:
                        scene.location = line_bits[0][prefix_len:]
                        scene.description = line_bits[1]
                    else:
                        scene.location = line[prefix_len:]
                    episode.scenes.append(scene)

                # otherwise line is either dialog or context_info within a scene
                else:
                    line_bits = line.split(': ', 1)
                    # if line contains a colon, we assume it splits a character name and their dialog
                    if len(line_bits) > 1:
                        dialog = SceneDialog()
                        dialog.spoken_by = line_bits[0]
                        # if line contains substrings wrapped in brackets or parens, extract these into context_info
                        extract_context_info_from_dialog(line_bits[1], dialog)
                        scene.events.append(dialog)
                    # otherwise assume the line is context_info between lines of dialog within a scene
                    else:
                        event = SceneEvent()
                        if line[0] == '[':
                            line = line[1:-1]
                        event.context_info = line
                        scene.events.append(event)

            # clean up: if first event in scene is nothing but context_info, move it to the scene.description field and delete the event
            for scene in episode.scenes:
                if not scene.description and len(scene.events) > 0 and isinstance(scene.events[0], SceneEvent) and scene.events[0].context_info:
                    scene.description = scene.events[0].context_info
                    scene.events = scene.events[1:]

        print('----------------------------------------------------------------------------')
        print(f'len(episode.scenes)={len(episode.scenes)}')

        return episode
    

'''
If line contains substrings wrapped in brackets or parens, extract these into context_info
'''
def extract_context_info_from_dialog(raw_line: str, dialog: SceneDialog) -> None:
    ctx_info = re.findall(r'[\(\[].*?[\)\]]', raw_line)
    if len(ctx_info) == 0:
        dialog.line = raw_line
    else:
        dialog.line = re.sub("[\(\[].*?[\)\]]", "", raw_line)
        dialog.context_info = ''
        for ci in ctx_info:
            dialog.context_info = f'{dialog.context_info}{ci[1:-1]} '

        dialog.line = dialog.line.strip()
        dialog.context_info = dialog.context_info.strip()
