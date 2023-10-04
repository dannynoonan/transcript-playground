from enum import Enum
from pydantic import BaseModel


class ShowKey(str, Enum):
    TNG = "TNG"
    GoT = "GoT"
    Succession = "Succession"


# class TranscriptType(str, Enum):
#     Fanon = "Fanon"
#     TOC = "TOC"
#     Default = "Default"
#     ALL = "ALL"


class Status(BaseModel):
    message: str


show_metadata = {
    'GoT': {
        'full_name': 'Game of Thrones',
        'wikipedia_label': 'List_of_Game_of_Thrones_episodes',
        'show_transcripts_domain': 'https://gameofthronesfanon.fandom.com/',
        'listing_url': 'wiki/Category:Transcripts',
        'episode_subdir': '/wiki/',
        'transcript_type_match_strings': [
            '_(Fanon)/Transcript',
            '/Transcript'
        ],
        # 'transcript_types': {
        #     'TOC': '/Transcript',
        #     'Fanon': '_(Fanon)/Transcript',
        # }
    },
    'TNG': {
        'full_name': 'Star Trek: The Next Generation',
        'wikipedia_label': 'List_of_Star_Trek:_The_Next_Generation_episodes',
        'show_transcripts_domain': 'http://www.chakoteya.net/NextGen/',
        'listing_url': 'episodes.htm',
        'episode_subdir': '',
        'transcript_type_match_strings': [
            'Default'
        ],
        # 'transcript_types': {
        #     'Default': 'Default'
        # }
    }
}


WIKIPEDIA_DOMAIN = 'https://en.wikipedia.org/wiki/'

GOT_SCENE_CHANGE_PREFIXES = ['CUT TO: ', 'EXT. ', 'INT. ']
TNG_CAPTAINS_LOG_PREFIX = "Captain's log"
