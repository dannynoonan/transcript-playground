from bs4 import BeautifulSoup

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

            # episode_open_prefixes = ['[Episode opens ', '[Episodes opens ', 'TITLE SEQUENCE']
            scene_change_prefixes = ['CUT TO: ', 'EXT. ']

            first_line = True
            for line in transcript_lines:
                line = line.get_text()
                line = line.rstrip()

                if first_line:
                    first_line = False
                    scene = Scene()
                    scene.description = line
                    episode.scenes.append(scene)

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

                else:
                    line_bits = line.split(': ', 1)
                    if len(line_bits) > 1:
                        dialog = SceneDialog()
                        dialog.spoken_by = line_bits[0]
                        dialog.line = line_bits[1]
                        scene.events.append(dialog)
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

        return episode