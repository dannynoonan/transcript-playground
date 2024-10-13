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
