from enum import Enum
# from pydantic import BaseModel


class ShowKey(str, Enum):
    TNG = "TNG"
    GoT = "GoT"
    Succession = "Succession"


# class TranscriptType(str, Enum):
#     Fanon = "Fanon"
#     TOC = "TOC"
#     Default = "Default"
#     ALL = "ALL"


# class Status(BaseModel):
#     message: str


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
        'regular_cast': {
            'PICARD': {'color': 'CornflowerBlue'},
            'RIKER': {'color': 'Burlywood'},
            'WORF': {'color': 'Crimson'}, 
            'DATA': {'color': 'Chartreuse'},
            'LAFORGE': {'color': 'Coral'},
            'TROI': {'color': 'Cyan'},
            'CRUSHER': {'color': 'DarkGoldenrod'},
            'COMPUTER': {'color': 'CadetBlue'},
            'WESLEY': {'color': 'DarkCyan'},
            "O'BRIEN": {'color': 'Cornsilk'},
            'GUINAN': {'color': 'MediumVioletRed'},
            'TASHA': {'color': 'MediumSlateBlue'},
            'PULASKI': {'color': 'RebeccaPurple'},
        },
        'recurring_cast': {
            'OGAWA': {'color': 'Thistle'},
            'Q': {'color': 'DarkKhaki'},
            'ALEXANDER': {'color': 'LightSalmon'}, 
            'RO': {'color': 'DeepPink'},
            'KEIKO': {'color': 'IndianRed'},
            'LWAXANA': {'color': 'Peru'},
            'BARCLAY': {'color': 'DodgerBlue'},
            'LORE': {'color': 'Aquamarine'},
            'GOWRON': {'color': 'DarkSeaGreen'},
            'RAGER': {'color': 'LawnGreen'},
            'FELTON': {'color': 'SlateBlue'},
            'GATES': {'color': 'DarkOrange'},
        },
    }
}

EXTRA_SPEAKER_COLORS = ['PaleVioletRed', 'Tomato', 'Magenta', 'LightGreen', 'SteelBlue', 'Bisque', 'LightCoral', 'HotPink', 'Gold',
                        'BlueViolet', 'PaleGreen', 'Aqua', 'RosyBrown', 'FireBrick', 'Indigo', 'Olive', 'DeepSkyBlue', 'Maroon',
                        'PeachPuff', 'Orchid', 'ForestGreen', 'LightBlue', 'Tan', 'Violet', 'Orange', 'Purple', 'Chocolate',
                        'OrangeRed', 'PapayaWhip', 'DarkSlateBlue', 'MediumSeaGreen', 'DarkOliveGreen', 'PowderBlue', 'Sienna']

TOPIC_COLORS = {
    'Action': 'DarkGoldenrod',
    'Comedy': 'Crimson',
    'Horror': 'MediumSeaGreen',
    'Drama': 'Fuchsia',
    'SciFi': 'DeepSkyBlue',
    'Fantasy': 'Orange',
    'Thriller': 'MediumBlue',
    'Crime': 'Maroon',
    'War': 'Turquoise',
    'Musical': 'SlateBlue',
    'Romance': 'Coral',
    'Western': 'BurlyWood',
    'Historical': 'LightSlateGray',
    'Sports': 'SpringGreen',
}


# TODO needs to be an algorithm
BGCOLORS_TO_TEXT_COLORS = {
    'DarkGoldenrod': 'Black',
    'Crimson': 'White',
    'MediumSeaGreen': 'Black',
    'Fuchsia': 'White',
    'DeepSkyBlue': 'Black',
    'Orange': 'Black',
    'MediumBlue': 'White',
    'Maroon': 'White',
    'Turquoise': 'Black',
    'SlateBlue': 'White',
    'Coral': 'Black',
    'BurlyWood': 'Black',
    'LightSlateGray': 'Black',
    'SpringGreen': 'White',
}


WIKIPEDIA_DOMAIN = 'https://en.wikipedia.org/wiki/'

GOT_SCENE_CHANGE_PREFIXES = ['CUT TO: ', 'EXT. ', 'INT. ']
TNG_CAPTAINS_LOG_PREFIX = "captain's log"

EPISODE_TOPIC_GROUPINGS = ['universalGenres', 'focusedGpt35_TNG', 'universalGenresGpt35_v2']
SPEAKER_TOPIC_GROUPINGS = ['meyersBriggsKiersey', 'dndAlignments']

SPEAKERS_TO_IGNORE = ['_ALL_', 'ALL', 'BOTH']


# TODO 9/21/24 verify that this is unused, and should it be removed? it head-faked me into thinking it was active
# show_ontology = {
#     'TNG': {
#         'characters': {
#             'PICARD': {
#                 'source_alts': ['JEAN-LUC'],
#                 'actor_names': ['Patrick Stewart', 'Patrick Stuart'],
#                 'external_errs': ['Jean Luc', 'Jeanluc', 'John luke', 'Pickard'],
#                 'external_alts': ['Captain'],
#                 'internal_alts': ['Captain']
#             },
#             'RIKER': {
#                 'source_alts': [],
#                 'actor_names': ['Jonathan Frakes', 'Johnathan Frakes'],
#                 'external_errs': ['Ryker'],
#                 'external_alts': ['number one'],
#                 'internal_alts': ['Commander', 'number one']
#             },
#             'CRUSHER': {
#                 'source_alts': ['BEVERLY'],
#                 'actor_names': ['Gates McFadden'],
#                 'external_errs': [],
#                 'external_alts': ['Doctor', 'Medical Officer'],
#                 'internal_alts': ['Doctor', 'Chief Medical Officer']
#             },
#             'DATA': {
#                 'source_alts': [],
#                 'actor_names': ['Brent Spiner'],
#                 'external_errs': [],
#                 'external_alts': ['android'],
#                 'internal_alts': ['Lieutenant Commander', 'android']
#             },
#             'LAFORGE': {
#                 'source_alts': ['GEORDI'], 
#                 'actor_names': ['LeVar Burton', 'Lavar Burton'], 
#                 'external_errs': ['Jordy', 'Giordi'],
#                 'external_alts': ['Chief Engineer', 'La Forge'],
#                 'internal_alts': ['Chief Engineer', 'La Forge']
#             },
#             'WORF': {
#                 'source_alts': [],
#                 'actor_names': ['Michael Dorn', 'Michael Dorne'],
#                 'external_errs': ['clingon', 'cling-on', 'cling on'],
#                 'external_alts': ['klingon', 'Security Officer'],
#                 'internal_alts': ['Chief Security Officer']
#             },
#             'TROI': {
#                 'source_alts': [],
#                 'actor_names': ['Marina Sirtis'], 
#                 'external_errs': ['Troy', 'Deana', 'Dianna'],
#                 'external_alts': ['Counselor', 'Deanna'],
#                 'internal_alts': ['Counselor', 'Deanna']
#             },
#             'WESLEY': {
#                 'source_alts': [],
#                 'actor_names': ['Wil Wheaton', 'Will Wheaton'],
#                 'external_errs': [],
#                 'external_alts': ['Ensign Crusher', 'Wes'],
#                 'internal_alts': ['Ensign Crusher', 'Wes']
#             },
#             'PULASKI': {
#                 'source_alts': [],
#                 'actor_names': ['Diana Muldaur'],
#                 'external_errs': ['Catherine', ],
#                 'external_alts': ['Doctor', 'Medical Officer', 'Katherine'],
#                 'internal_alts': ['Doctor', 'Chief Medical Officer', 'Katherine']
#             },
#             "O'BRIEN": {
#                 'source_alts': ['MILES'], 
#                 'actor_names': ['Colm Meaney', 'Colm Meany'],
#                 'external_errs': ['Obrien'],
#                 'external_alts': ['Chief Engineer'],
#                 'internal_alts': ['Chief Engineer']
#             },
#             'TASHA': {
#                 'source_alts': [],
#                 'actor_names': ['Denise Crosby'],
#                 'external_errs': [],
#                 'external_alts': ['Natasha', 'Yar'],
#                 'internal_alts': ['Security Officer', 'Natasha', 'Yar']
#             },
#             'GUINAN': {
#                 'source_alts': [],
#                 'actor_names': ['Whoopi Goldberg', 'Whoopie Goldberg'],
#                 'external_errs': ['Guynan'],
#                 'external_alts': [],
#                 'internal_alts': ['bartender']
#             },
#             'OGAWA': {
#                 'source_alts': [],
#                 'actor_names': ['Patti Yasutake', 'Patty Yasutake', 'Pattie Yasutake'],
#                 'external_errs': [],
#                 'external_alts': ['nurse'],
#                 'internal_alts': ['Nurse', 'Alyssa']
#             },
#             'Q': {
#                 'source_alts': [],
#                 'actor_names': ['John de Lancie', 'John Delancey', 'John Delancy', 'Jon de Lancie', 'Jon Delancey', 'Jon Delancy'],
#                 'external_errs': [],
#                 'external_alts': ['omnipotent', 'god-like'],
#                 'internal_alts': ['omnipotent', 'consortium']
#             }
#         },
#         'things_repl': {
#             'Forengi': 'Ferengi',
#         },
#         'things_supp': {
#             'Enterprise': 'Starship',
#             'Vulcan': 'alien logic species',
#             'Klingon': 'alien warrior species',
#             'Ferengi': 'alien species',
#             'Romulan': 'alien species',
#             'Cardassian': 'alien species',
#             'Bajoran': 'alien species',
#         }
#     },
#     'GoT': {
#         'characters': {},
#         'things_repl': {},
#         'things_supp': {}
#     }
# }


# def build_query_supplement_map(show_key: str) -> dict:
#     qsm = {}
#     show_char_meta = show_ontology[show_key]['characters']
#     for char, meta in show_char_meta.items():
#         char = char.lower()
#         for w in meta['source_alts']:
#             w = w.lower()
#             if w in qsm:
#                 qsm[w].append(char)
#             else:
#                 qsm[w] = [char]
#         for w in meta['external_alts']:
#             w = w.lower()
#             if w in qsm:
#                 qsm[w].append(char)
#             else:
#                 qsm[w] = [char]
#     # TODO should this, or even the whole function, just fold into query_expansion_map?
#     show_thing_meta = show_ontology[show_key]['things_supp']
#     for w, supp in show_thing_meta.items():
#         w = w.lower()
#         supp = supp.lower()
#         if w in qsm:
#             qsm[w].append(supp)
#         else:
#             qsm[w] = [supp]
#     return qsm


# def build_query_replacement_map(show_key: str) -> dict:
#     qrm = {}
#     show_char_meta = show_ontology[show_key]['characters']
#     for char, meta in show_char_meta.items():
#         char = char.lower()
#         for w in meta['actor_names']:
#             w = w.lower()
#             if w in qrm:
#                 qrm[w].append(char)
#             else:
#                 qrm[w] = [char]
#         for w in meta['external_errs']:
#             w = w.lower()
#             if w in qrm:
#                 qrm[w].append(char)
#             else:
#                 qrm[w] = [char]
#     show_thing_meta = show_ontology[show_key]['things_repl']
#     for w, repl in show_thing_meta.items():
#         w = w.lower()
#         repl = repl.lower()
#         if w in qrm:
#             qrm[w].append(repl)
#         else:
#             qrm[w] = [repl]
#     return qrm
            
    
# def build_query_expansion_map(show_key: str) -> dict:
#     qem = {}
#     show_char_meta = show_ontology[show_key]['characters']
#     for char, meta in show_char_meta.items():
#         qem[char.lower()] = [int_alt.lower() for int_alt in meta['source_alts']]
#     for char, meta in show_char_meta.items():
#         qem[char.lower()].extend([int_alt.lower() for int_alt in meta['internal_alts']])
#     return qem
