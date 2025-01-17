import dash_bootstrap_components as dbc
from dash import dash_table
import pandas as pd

import app.fig_meta.color_meta as cm
import app.page_builder_service.page_components as pc


def generate_character_dropdown_options(show_key: str, all_characters: list) -> list:
    character_dropdown_options = []
    for character_key in all_characters:
        character_dropdown_options.append({'label': character_key, 'value': character_key})
    
    return character_dropdown_options
