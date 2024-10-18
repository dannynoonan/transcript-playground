import plotly.graph_objects as go


FRAME_RATE = 1000


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


def blank_fig():
    '''
    Best I've come up with so far to dynamically show or hide (almost) a graph that's declared as a dash page object
    '''
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None, width=10, height=10)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    
    return fig


def apply_animation_settings(fig: go.Figure, base_fig_title: str, frame_rate: int = None) -> None:
    """
    generic recipe of steps to execute on animation figure after its built: explicitly set frame rate, dynamically update fig title, etc
    """

    print(f'in apply_animation_settings frame_rate={frame_rate}')
    if not frame_rate:
        frame_rate = FRAME_RATE

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_rate
        
    for button in fig.layout.updatemenus[0].buttons:
        button["args"][1]["frame"]["redraw"] = True
    
    # first_step = True
    for step in fig.layout.sliders[0].steps:
        step["args"][1]["frame"]["redraw"] = True
        # if first_step:
        #     # step["args"][1]["frame"]["duration"] = 0
        #     first_step = False
        # step["args"][1]["frame"]["duration"] = frame_rate

    # for k in range(len(fig.frames)):
    #     year = YEAR_0 + (k*4)
    #     era = get_era_for_year(year)
    #     fig.frames[k]['layout'].update(title_text=f'{base_fig_title}: {year} ({era})')
        
    print(f'fig.layout={fig.layout}')
