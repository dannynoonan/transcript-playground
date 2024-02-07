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


colors = ["cornflowerblue", "burlywood", "crimson", "cadetblue", "coral", "chartreuse", "cornsilk", "cyan", "darkgoldenrod", "darkcyan"]


episode_keep_cols = ['title', 'season', 'sequence_in_season', 'air_date', 'focal_speakers', 'focal_locations', 'scene_count', 'episode_key']
episode_drop_cols = ['doc_id', 'show_key', 'indexed_ts']
cluster_cols = ['Cluster', 'cluster_color']
