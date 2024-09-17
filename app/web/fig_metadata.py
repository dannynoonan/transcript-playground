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


colors = ["cornflowerblue", "burlywood", "crimson", "chartreuse", "coral", "cyan", "darkgoldenrod", "cadetblue", "darkcyan", "cornsilk"]
colors = colors + colors + colors + colors
text_colors = ["white", "black", "white", "black", "white", "black", "white", "white", "white", "black"]
text_colors = text_colors + text_colors + text_colors + text_colors

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
    [(-0.4, 0), (0, 0), (0.4, 0)], #3
    [(-0.2, 0.2), (0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)], #4 2,2
    [(-0.4, 0.2), (0, 0.2), (0.4, 0.2), (-0.2, -0.2), (0.2, -0.2)], #5 3,2
    [(-0.4, 0.2), (0, 0.2), (0.4, 0.2), (-0.4, -0.2), (0, -0.2), (0.4, -0.2)], #6 3,3
    [(-0.4, 0.4), (0, 0.4), (0.4, 0.4), (-0.2, 0), (0.2, 0), (-0.2, -0.4), (0.2, -0.4)], #7 3,2,2
    [(-0.4, 0.4), (0, 0.4), (0.4, 0.4), (-0.2, 0), (0.2, 0), (-0.4, -0.4), (0, -0.4), (0.4, -0.4)], #8 3,2,3
    [(-0.4, 0.4), (0, 0.4), (0.4, 0.4), (-0.4, 0), (0, 0), (0.4, 0), (-0.4, -0.4), (0, -0.4), (0.4, -0.4)], #9 3,3,3
]


mbti_types = {}
mbti_types['ESFJ'] = dict(coords=(0,1), descr='Provider')
mbti_types['ESFP'] = dict(coords=(0,2), descr='Performer')
mbti_types['ISFJ'] = dict(coords=(1,1), descr='Protector')
mbti_types['ISFP'] = dict(coords=(1,2), descr='Composer')
mbti_types['ENFP'] = dict(coords=(0,3), descr='Champion')
mbti_types['ENFJ'] = dict(coords=(0,4), descr='Teacher')
mbti_types['INFP'] = dict(coords=(1,3), descr='Healer')
mbti_types['INFJ'] = dict(coords=(1,4), descr='Counselor')
mbti_types['ISTJ'] = dict(coords=(2,1), descr='Inspector')
mbti_types['ISTP'] = dict(coords=(2,2), descr='Operator')
mbti_types['ESTJ'] = dict(coords=(3,1), descr='Supervisor')
mbti_types['ESTP'] = dict(coords=(3,2), descr='Promoter')
mbti_types['INTP'] = dict(coords=(2,3), descr='Architect')
mbti_types['INTJ'] = dict(coords=(2,4), descr='Mastermind')
mbti_types['ENTP'] = dict(coords=(3,3), descr='Inventor')
mbti_types['ENTJ'] = dict(coords=(3,4), descr='Field Marshall')


dnda_types = {}
dnda_types['Chaotic.Evil'] = dict(coords=(0,1), descr='Chaotic Evil')
dnda_types['Chaotic.Neutral'] = dict(coords=(1,1), descr='Chaotic Neutral')
dnda_types['Chaotic.Good'] = dict(coords=(2,1), descr='Chaotic Good')
dnda_types['Neutral.Evil'] = dict(coords=(0,2), descr='Neutral Evil')
dnda_types['Neutral.Neutral'] = dict(coords=(1,2), descr='Neutral')
dnda_types['Neutral.Good'] = dict(coords=(2,2), descr='Neutral Good')
dnda_types['Lawful.Evil'] = dict(coords=(0,3), descr='Lawful Evil')
dnda_types['Lawful.Neutral'] = dict(coords=(1,3), descr='Lawful Neutral')
dnda_types['Lawful.Good'] = dict(coords=(2,3), descr='Lawful Good')
