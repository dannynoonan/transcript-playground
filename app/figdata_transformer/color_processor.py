from PIL import ImageColor

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
    