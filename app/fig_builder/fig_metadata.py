from app.es.es_metadata import SENTIMENT_FIELDS, TOPICS_FIELDS, FOCAL_FIELDS


class FigDimensions():
    def __init__(self):
        self.MD5 = 650
        self.MD6 = 788
        self.MD7 = 924
        self.MD8 = 1080
        self.MD10 = 1320
        self.MD11 = 1450
        self.MD12 = 1610

    def square(self, width):
        return width

    def hdef(self, width):
        return width * .562
    
    def crt(self, width):
        return width * .75

    def wide_door(self, width):
        return width * 1.25

    def narrow_door(self, width):
        return width * 1.5
    

fig_dims = FigDimensions()


# TODO clarify the scope of the usage of this 
colors = ["cornflowerblue", "burlywood", "crimson", "chartreuse", "coral", "cyan", "darkgoldenrod", "cadetblue", "darkcyan", "cornsilk"]
text_colors = ["white", "black", "white", "black", "white", "black", "white", "white", "white", "black"]
text_color_map = {
    "cornflowerblue": "white",
    "burlywood": "black", 
    "crimson": "white", 
    "chartreuse": "black", 
    "coral": "white", 
    "cyan": "black", 
    "darkgoldenrod": "white", 
    "cadetblue": "white", 
    "darkcyan": "white", 
    "cornsilk": "black"
}

color_map = {}
for v in colors:
    color_map[v] = v


# TODO happened upon these in Aug 2024, not 100% of their usage but I suspect es query response filters/flags would handle this better
# episode_keep_cols = ['title', 'season', 'sequence_in_season', 'air_date', 'focal_speakers', 'focal_locations', 'scene_count', 'episode_key']
episode_keep_cols = ['title', 'season', 'sequence_in_season', 'air_date', 'scene_count', 'episode_key']
# episode_drop_cols = ['doc_id', 'show_key', 'indexed_ts', 'topics_universal', 'topics_focused', 'topics_universal_tfidf', 'topics_focused_tfidf']
episode_drop_cols = ['doc_id', 'show_key', 'indexed_ts'] + SENTIMENT_FIELDS + TOPICS_FIELDS + FOCAL_FIELDS
cluster_cols = ['cluster', 'cluster_color']


topic_grid_coord_deltas = [
    [], #0
    [(0, 0)], #1
    [(-0.2, 0), (0.2, 0)], #2 
    [(-0.3, 0), (0, 0), (0.3, 0)], #3
    [(-0.2, 0.2), (0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)], #4 2,2
    [(-0.3, 0.2), (0, 0.2), (0.3, 0.2), (-0.2, -0.2), (0.2, -0.2)], #5 3,2
    [(-0.3, 0.2), (0, 0.2), (0.3, 0.2), (-0.3, -0.2), (0, -0.2), (0.3, -0.2)], #6 3,3
    [(-0.3, 0.3), (0, 0.3), (0.3, 0.3), (-0.2, 0), (0.2, 0), (-0.2, -0.3), (0.2, -0.3)], #7 3,2,2
    [(-0.3, 0.3), (0, 0.3), (0.3, 0.3), (-0.2, 0), (0.2, 0), (-0.3, -0.3), (0, -0.3), (0.3, -0.3)], #8 3,2,3
    [(-0.3, 0.3), (0, 0.3), (0.3, 0.3), (-0.3, 0), (0, 0), (0.3, 0), (-0.3, -0.3), (0, -0.3), (0.3, -0.3)], #9 3,3,3
]


mbti_types = {
     # SF
    'ESFJ': {
        'descr': 'ESFJ: Provider',
        'color': 'Orange',
        'coords': [0, 1, 0, 1]
    },
    'ESFP': {
        'descr': 'ESFP: Performer',
        'color': 'Orange',
        'coords': [0, 1, 1, 2]
    },
    'ISFJ': {
        'descr': 'ISFJ: Protector',
        'color': 'Orange',
        'coords': [1, 2, 0, 1]
    },
    'ISFP': {
        'descr': 'ISFP: Composer',
        'color': 'Orange',
        'coords': [1, 2, 1, 2]
    },
    # NF
    'ENFP': {
        'descr': 'ENFP: Champion',
        'color': 'YellowGreen',
        'coords': [0, 1, 2, 3]
    },
    'ENFJ': {
        'descr': 'ENFJ: Teacher',
        'color': 'YellowGreen',
        'coords': [0, 1, 3, 4]
    },
    'INFP': {
        'descr': 'INFP: Healer',
        'color': 'YellowGreen',
        'coords': [1, 2, 2, 3]
    },
    'INFJ': {
        'descr': 'INFJ: Counselor',
        'color': 'YellowGreen',
        'coords': [1, 2, 3, 4]
    },
    # ST
    'ISTJ': {
        'descr': 'ISTJ: Inspector',
        'color': 'Crimson',
        'coords': [2, 3, 0, 1]
    },
    'ISTP': {
        'descr': 'ISTP: Operator',
        'color': 'Crimson',
        'coords': [2, 3, 1, 2]
    },
    'ESTJ': {
        'descr': 'ESTJ: Supervisor',
        'color': 'Crimson',
        'coords': [3, 4, 0, 1]
    },
    'ESTP': {
        'descr': 'ESTP: Promoter',
        'color': 'Crimson',
        'coords': [3, 4, 1, 2]
    },
    # NT
    'INTP': {
        'descr': 'INTP: Architect',
        'color': 'MediumAquamarine',
        'coords': [2, 3, 2, 3]
    },
    'INTJ': {
        'descr': 'INTJ: Mastermind',
        'color': 'MediumAquamarine',
        'coords': [2, 3, 3, 4]
    },
    'ENTP': {
        'descr': 'ENTP: Inventor',
        'color': 'MediumAquamarine',
        'coords': [3, 4, 2, 3]
    },
    'ENTJ': {
        'descr': 'ENTJ: Field Marshall',
        'color': 'MediumAquamarine',
        'coords': [3, 4, 3, 4]
    }
}


dnda_types = {
     # SF
    'Chaotic.Evil': {
        'descr': 'Chaotic Evil',
        'color': 'Red',
        'coords': [0, 1, 0, 1]
    },
    'Chaotic.Neutral': {
        'descr': 'Chaotic Neutral',
        'color': 'Purple',
        'coords': [1, 2, 0, 1]
    },
    'Chaotic.Good': {
        'descr': 'Chaotic.Good',
        'color': 'Blue',
        'coords': [2, 3, 0, 1]
    },
    'Neutral.Evil': {
        'descr': 'Neutral Evil',
        'color': 'Orange',
        'coords': [0, 1, 1, 2]
    },
    'Neutral.Neutral': {
        'descr': 'Neutral',
        'color': 'Gray',
        'coords': [1, 2, 1, 2]
    },
    'Neutral.Good': {
        'descr': 'Neutral Good',
        'color': 'LightSeaGreen',
        'coords': [2, 3, 1, 2]
    },
    'Lawful.Evil': {
        'descr': 'Lawful Evil',
        'color': 'Yellow',
        'coords': [0, 1, 2, 3]
    },
    'Lawful.Neutral': {
        'descr': 'Lawful Neutral',
        'color': 'GreenYellow',
        'coords': [1, 2, 2, 3]
    },
    'Lawful.Good': {
        'descr': 'Lawful Good',
        'color': 'Green',
        'coords': [2, 3, 2, 3]
    }
}
