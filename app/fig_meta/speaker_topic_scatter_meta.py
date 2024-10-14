

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


def to_mbti_x(topic_key: str):
    if 'ESF' in topic_key or 'ENF' in topic_key:
        return 0.5
    if 'ISF' in topic_key or 'INF' in topic_key:
        return 1.5
    if 'IST' in topic_key or 'INT' in topic_key:
        return 2.5
    if 'EST' in topic_key or 'ENT' in topic_key:
        return 3.5
        

def to_mbti_y(topic_key: str):
    if 'SFJ' in topic_key or 'STJ' in topic_key:
        return 0.5
    if 'SFP' in topic_key or 'STP' in topic_key:
        return 1.5
    if 'NFP' in topic_key or 'NTP' in topic_key:
        return 2.5
    if 'NFJ' in topic_key or 'NTJ' in topic_key:
        return 3.5
    

def to_dnda_x(topic_key: str):
    if '.Evil' in topic_key:
        return 0.5
    if '.Neutral' in topic_key:
        return 1.5
    if '.Good' in topic_key:
        return 2.5
        

def to_dnda_y(topic_key: str):
    if 'Chaotic.' in topic_key:
        return 0.5
    if 'Neutral.' in topic_key:
        return 1.5
    if 'Lawful.' in topic_key:
        return 2.5


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
