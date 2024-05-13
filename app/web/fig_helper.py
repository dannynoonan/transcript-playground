import plotly.graph_objects as go

FRAME_RATE = 1000

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


def topic_cat_rank_color_mapper(topic_cat_i: int, hex_hue: int) -> str:
    i = topic_cat_i % 9
    j = int(topic_cat_i / 9)
    low = j
    high = 255 - j
    if i == 0:
        return f'rgb({hex_hue},{hex_hue},{high})'
    if i == 1:
        return f'rgb({hex_hue},{high},{hex_hue})'
    if i == 2:
        return f'rgb({high},{hex_hue},{hex_hue})'
    if i == 3:
        return f'rgb({hex_hue},{high},{high})'
    if i == 4:
        return f'rgb({high},{hex_hue},{high})'
    if i == 5:
        return f'rgb({high},{high},{hex_hue})'
    if i == 6:
        return f'rgb({hex_hue},{hex_hue},{low})'
    if i == 7:
        return f'rgb({hex_hue},{low},{hex_hue})'
    if i == 8:
        return f'rgb({low},{hex_hue},{hex_hue})'
    if i == 9:
        return f'rgb({hex_hue},{hex_hue},{hex_hue})'
    
    # if i == 7:
    #     return f'rgb({hex_hue},0,0)'
    # if i == 8:
    #     return f'rgb(0,{hex_hue},0)'
    # if i == 9:
    #     return f'rgb(0,0,{hex_hue})'

    # if i == 7:
    #     return f'rgb({hex_hue},0,255)'
    # if i == 8:
    #     return f'rgb(0,{hex_hue},255)'
    # if i == 9:
    #     return f'rgb(0,255,{hex_hue})'
    # if i == 10:
    #     return f'rgb({hex_hue},255,0)'
    # if i == 11:
    #     return f'rgb(255,{hex_hue},0)'
    # if i == 12:
    #     return f'rgb(255,0,{hex_hue})'
        