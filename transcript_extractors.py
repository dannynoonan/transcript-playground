from bs4 import BeautifulSoup
import re

from type_system import Episode, Scene, SceneEvent, SceneDialog


class TranscriptExtractor(object):
    def __init__(self, show_key: str, show_meta: dict, transcript_type: str, transcript_url: str):
        self.show_key = show_key
        self.show_meta = show_meta
        self.transcript_type = transcript_type
        self.transcript_url = transcript_url

    def extract_transcript(self, transcript_soup: BeautifulSoup):
        episode = Episode()

        # extraction biz logic by show / transcript_type 
        # GoT
        if self.show_key == 'GoT':
            # Fanon
            if self.transcript_type == 'Fanon':
                episode.show = self.show_key
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
                        add_dialog_or_event(line, scene)

                # clean up: if first event in scene is nothing but context_info, move it to the scene.description field and delete the event
                for scene in episode.scenes:
                    if not scene.description and len(scene.events) > 0 and isinstance(scene.events[0], SceneEvent) and scene.events[0].context_info:
                        scene.description = scene.events[0].context_info
                        scene.events = scene.events[1:]

            # extract title from url (temporary hack)
            episode.transcript_url = self.transcript_url
            title = self.transcript_url.removeprefix(self.show_meta['show_transcripts_domain'])
            title = title.removeprefix(self.show_meta['episode_subdir'])
            transcript_type_string_match = self.show_meta['transcript_types'][self.transcript_type]
            title = title.replace(transcript_type_string_match, '')
            episode.title = title

        elif self.show_key == 'TNG':
            episode.show = self.show_key
            transcript_file = self.transcript_url.split('/')[-1]
            episode.title = transcript_file.split('.')[0]
            episode.air_date = ''

            transcript_raw = transcript_soup.find_all('table')[0]
            first_scene_location = transcript_raw.find_all('b')[0].get_text()
            scene_locations_and_content = transcript_raw.find_all('p')

            captains_log_prefix = "Captain's log"

            scene = Scene()
            scene.location = first_scene_location[1:-1]
            episode.scenes.append(scene)

            # is_location = False
            for slac in scene_locations_and_content:
                locations = slac.find_all('b')
                if locations and len(locations) > 0:
                    scene = Scene()
                    location = locations[0].get_text()
                    scene.location = location[1:-1]
                    episode.scenes.append(scene)
                else:
                    lines = [line for line in slac.strings]
                    # lines = slac.find_all('br')
                    for line in lines:
                        line = line.get_text()

                        # preliminary trimming
                        line = line.replace('\n', ' ')
                        line = line.replace('\r', ' ')
                        line = line.replace('  ', ' ')
                        line = line.strip()

                        # if line starts with captains_log_prefix, prepend line with Pickard's name and handle it as normal dialog
                        if line.startswith(captains_log_prefix):
                            line = f'PICARD: {line}'

                        add_dialog_or_event(line, scene)

        print('----------------------------------------------------------------------------')
        print(f'len(episode.scenes)={len(episode.scenes)}')

        return episode
    

'''
Parse line text to extract sceneDialog and sceneEvent info to be added to scene
'''
def add_dialog_or_event(line: str, scene: Scene):
    line_bits = line.split(': ', 1)
    # if line contains a colon, we assume it splits a character name and their dialog
    if len(line_bits) > 1:
        dialog = SceneDialog()
        # if line contains substrings wrapped in brackets or parens, extract these into context_info
        extract_context_info_from_dialog(line_bits[0], dialog, 'spoken_by')
        extract_context_info_from_dialog(line_bits[1], dialog, 'line')
        scene.events.append(dialog)
    # otherwise assume the line is context_info between lines of dialog within a scene
    else:
        event = SceneEvent()
        if line and line[0] in ['[', '(']:
            line = line[1:-1]
        event.context_info = line
        scene.events.append(event)


'''
If line contains substrings wrapped in brackets or parens, extract these into context_info
'''
def extract_context_info_from_dialog(raw_text: str, dialog: SceneDialog, field: str) -> None:
    ctx_info = re.findall(r'[\(\[].*?[\)\]]', raw_text)
    if len(ctx_info) == 0:
        processed_text = raw_text
    else:
        processed_text = re.sub("[\(\[].*?[\)\]]", "", raw_text)
        processed_text = processed_text.strip()
        if not dialog.context_info:
            dialog.context_info = ''
        for ci in ctx_info:
            dialog.context_info = f'{dialog.context_info}{ci[1:-1]} '
        dialog.context_info = dialog.context_info.strip()

    if field == 'spoken_by':
        dialog.spoken_by = processed_text
    else:
        dialog.line = processed_text
