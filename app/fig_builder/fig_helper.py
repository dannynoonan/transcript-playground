from PIL import ImageColor
import plotly.graph_objects as go

from app.show_metadata import show_metadata, EXTRA_SPEAKER_COLORS


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


def to_mbti_x(topic_key: str):
    if 'ESF' in topic_key or 'ISF' in topic_key:
        return 0.5
    if 'ISF' in topic_key or 'INF' in topic_key:
        return 1.5
    if 'INT' in topic_key or 'IST' in topic_key:
        return 2.5
    if 'ENT' in topic_key or 'EST' in topic_key:
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
    

def extract_parent(topic_key: str):
    topic_path = topic_key.split('.')
    return topic_path[0]


def flatten_topics(topics: list):
    out_list = []
    parents_seen = []
    for topic in topics:
        t_bits = topic['topic_key'].split('.')
        if len(t_bits) <= 1 or t_bits[0] in parents_seen:
            continue
        parents_seen.append(t_bits[0])
        out_list.append(topic['topic_key'])
    return ', '.join(out_list)


def build_and_annotate_scene_labels(fig: go.Figure, scenes: list) -> list:
    """
    helper function to layer scene labels into episode dialog gantt
    """

    # build markers and labels marking events 
    scene_lines = []
    yshift = -22 # NOTE might need to be derived based on speaker count / y axis length

    for scene in scenes:
        # add vertical line for each scene
        scene_line = dict(type='line', line_width=1, line_color='#A0A0A0', x0=scene['Start'], x1=scene['Start'], y0=0, y1=1, yref='paper')
        scene_lines.append(scene_line)
        # add annotation for each scene location
        fig.add_annotation(x=scene['Start'], y=0, text=scene['Task'], showarrow=False, 
            yshift=yshift, xshift=6, textangle=-90, align='left', yanchor='bottom',
            font=dict(family="Arial", size=10, color="#A0A0A0"))

    return scene_lines


def flatten_speaker_colors(show_key: str, to_rgb: bool = False) -> dict:
    speaker_colors = {}
    for s, d in show_metadata[show_key]['regular_cast'].items():
        if to_rgb:
            rgb = ImageColor.getcolor(d['color'], 'RGB')
            speaker_colors[s] = f'rgb{rgb}'.replace(' ', '')
        else:
            speaker_colors[s] = d['color']
    for s, d in show_metadata[show_key]['recurring_cast'].items():
        if to_rgb:
            rgb = ImageColor.getcolor(d['color'], 'RGB')
            speaker_colors[s] = f'rgb{rgb}'.replace(' ', '')
        else:
            speaker_colors[s] = d['color']
    return speaker_colors


def generate_speaker_color_discrete_map(show_key: str, speakers: list) -> dict:
    speaker_colors = flatten_speaker_colors(show_key)
    color_discrete_map = {}
    extra_speaker_i = 0
    for s in speakers:
        if s in speaker_colors:
            color_discrete_map[s] = speaker_colors[s]
        else:
            color_discrete_map[s] = EXTRA_SPEAKER_COLORS[extra_speaker_i]
            extra_speaker_i += 1
    return color_discrete_map


# def build_and_annotate_scene_blocks(scenes: list) -> list:
#     """
#     Helper function to layer scene blocks into episode dialog gantt
#     """

#     # build shaded blocks designating eras
#     scene_blocks = []

#     for scene in scenes:
#         # add rectangle for each era date range
#         block = dict(type='rect', line_width=1, x0=scene['Start'], x1=scene['Finish'], y0=0, y1=1,
#                      yref='paper',
#                     # fillcolor=era['color'], 
#                     opacity=0.12)
#         scene_blocks.append(block) 

#     return scene_blocks
        