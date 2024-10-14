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
