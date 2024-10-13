import matplotlib
import matplotlib.pyplot as plt
from PIL import ImageColor

import app.data_service.data_processor as dp
from app.show_metadata import show_metadata, EXTRA_SPEAKER_COLORS


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
            if extra_speaker_i >= len(EXTRA_SPEAKER_COLORS):
                extra_speaker_i = 0
            color_discrete_map[s] = EXTRA_SPEAKER_COLORS[extra_speaker_i]
            extra_speaker_i += 1

    return color_discrete_map


def matplotlib_gradient_to_rgb_strings(gradient_type: str):
    gradient = plt.cm.get_cmap(gradient_type)
    rgb_strings = []
    for c in gradient.colors:
        mpl_color = matplotlib.colors.to_rgb(c)
        rgb = tuple([int(c*255) for c in mpl_color])
        rgb_str = f'rgb{rgb}'.replace(' ', '')
        rgb_strings.append(rgb_str)
    return rgb_strings


def map_range_values_to_gradient(range_values: list, gradient_values: list) -> list:
    '''
    Both 'range_values' and 'gradient_values' are assumed to be sorted
    '''
    scaled_range_values = dp.scale_values(range_values, 0, len(gradient_values)-1)
    discrete_gradient_values = []
    for v in scaled_range_values:
        discrete_gradient_values.append(gradient_values[round(v)])

    # TODO incorporate Black vs White font color here?

    return discrete_gradient_values


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


# TODO clarify the scope of the usage of this
colors = ["cornflowerblue", "burlywood", "crimson", "chartreuse", "coral", "cyan", "darkgoldenrod", "cadetblue", "darkcyan", "cornsilk"]
text_colors = ["white", "black", "white", "black", "white", "black", "white", "white", "white", "black"]

color_map = {}
for v in colors:
    color_map[v] = v


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
    'Western': 'Burlywood',
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
    'Burlywood': 'Black',
    'LightSlateGray': 'Black',
    'SpringGreen': 'White',
    'CornflowerBlue': 'Black',
    'Chartreuse': 'Black',
    'Cyan': 'Black',
    'CadetBlue': 'Black',
    'DarkCyan': 'White',
    'Cornsilk': 'Black', 
    'MediumVioletRed': 'White',
    'MediumSlateBlue': 'White',
    'RebeccaPurple': 'White',
    'Thistle': 'Black',
    'DarkKhaki': 'White', 
    'LightSalmon': 'Black',
    'DeepPink': 'White',
    'IndianRed': 'Black',
    'Peru': 'Black',
    'DodgerBlue': 'Black',
    'Aquamarine': 'Black',
    'DarkSeaGreen': 'White',
    'LawnGreen': 'Black', 
    'DarkOrange': 'White',
    'PaleVioletRed': 'Black', 
    'Tomato': 'Black', 
    'Magenta': 'Black', 
    'LightGreen': 'Black', 
    'SteelBlue': 'White', 
    'Bisque': 'Black', 
    'LightCoral': 'Black', 
    'HotPink': 'Black', 
    'Gold': 'Black',
    'BlueViolet': 'White', 
    'PaleGreen': 'Black', 
    'Aqua': 'Black', 
    'RosyBrown': 'Black', 
    'FireBrick': 'Black', 
    'Indigo': 'White', 
    'Olive': 'Black', 
    'PeachPuff': 'Black',
    'Orchid': 'White', 
    'ForestGreen': 'White', 
    'LightBlue': 'Black', 
    'Tan': 'Black', 
    'Violet': 'White', 
    'Purple': 'White', 
    'Chocolate': 'White',
    'OrangeRed': 'White', 
    'PapayaWhip': 'Black', 
    'DarkSlateBlue': 'White', 
    'DarkOliveGreen': 'White', 
    'PowderBlue': 'Black', 
    'Sienna': 'Black',
    'Red': 'White'
}
    