# import dash
# from bertopic import BERTopic
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output, State
import os
import pandas as pd
import urllib.parse

import app.dash.components as cmp
# from app.dash import (
#     bertopic_model_clusters, episode_gantt_chart, location_line_chart, sentiment_line_chart, series_gantt_chart, series_search_results_gantt, 
#     show_3d_network_graph, speaker_3d_network_graph, show_cluster_scatter, show_network_graph, speaker_frequency_bar_chart, speaker_line_chart
# )
from app.dash_new import episode_palette
# import app.es.es_query_builder as esqb
# import app.es.es_response_transformer as esrt
import app.es.es_read_router as esr
# import app.nlp.embeddings_factory as ef
from app.nlp.nlp_metadata import BERTOPIC_DATA_DIR, BERTOPIC_MODELS_DIR, OPENAI_EMOTIONS
from app.show_metadata import ShowKey
# import app.utils as utils
import app.web.fig_builder as fb
# import app.web.fig_metadata as fm


dapp_new = Dash(__name__,
            external_stylesheets=[dbc.themes.SOLAR],
            requests_pathname_prefix='/tsp_dash_new/')

# app layout
dapp_new.layout = dbc.Container(fluid=True, children=[
    cmp.url_bar_and_content_div,
])


# Index callbacks
@dapp_new.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    Input('url', 'search'))
def display_page(pathname, search):
    # parse params
    parsed = urllib.parse.urlparse(search)
    parsed_dict = urllib.parse.parse_qs(parsed.query)
    print(f'parsed_dict={parsed_dict}')

    if pathname == "/tsp_dash_new/episode-palette":
         # generate form-backing data 
        all_simple_episodes = esr.fetch_simple_episodes(ShowKey('TNG'))
        episode_dropdown_options = []
        for episode in all_simple_episodes['episodes']:
            label = f"{episode['title']} (S{episode['season']}:E{episode['sequence_in_season']})"
            episode_dropdown_options.append({'label': label, 'value': episode['episode_key']})

        # parse episode_key from params
        if 'episode_key' in parsed_dict:
            episode_key = parsed_dict['episode_key']
            if isinstance(episode_key, list):
                episode_key = episode_key[0]

        speaker_episodes_response = esr.fetch_speakers_for_episode(ShowKey('TNG'), episode_key)
        episode_speakers = speaker_episodes_response['speaker_episodes']
        speaker_dropdown_options = [s['speaker'] for s in episode_speakers]
        return episode_palette.generate_content(episode_dropdown_options, episode_key, speaker_dropdown_options)
    


############ episode-palette callbacks
@dapp_new.callback(
    Output('episode-dialog-timeline-new', 'figure'),
    Output('episode-location-timeline-new', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'))    
def render_episode_chromatographs(show_key: str, episode_key: str):
    print(f'in render_episode_chromatographs, show_key={show_key} episode_key={episode_key}')

    # generate data and build episode gantt charts
    response = esr.generate_episode_gantt_sequence(ShowKey(show_key), episode_key)
    episode_dialog_timeline = fb.build_episode_gantt(show_key, response['dialog_timeline'])
    episode_location_timeline = fb.build_episode_gantt(show_key, response['location_timeline'])

    return episode_dialog_timeline, episode_location_timeline


############ sentiment-line-chart callbacks
@dapp_new.callback(
    Output('sentiment-line-chart-new', 'figure'),
    # Output('episode-speaker-options', 'options'),
    Input('show-key2', 'value'),
    Input('episode-key2', 'value'),
    Input('freeze-on', 'value'),
    Input('emotion', 'value'),
    Input('speaker', 'value'))    
def render_episode_sentiment_line_chart_new(show_key: str, episode_key: str, freeze_on: str, emotion: str, speaker: str):
    print(f'in render_episode_sentiment_line_chart, show_key={show_key} episode_key={episode_key} freeze_on={freeze_on} emotion={emotion} speaker={speaker}')

    if freeze_on == 'emotion':
        emotions = [emotion]
        # speakers = ['PICARD', 'RIKER', 'DATA', 'TROI', 'LAFORGE', 'WORF', 'CRUSHER']
        speakers = []
        focal_property = 'speaker'
    elif freeze_on == 'speaker':
        emotions = OPENAI_EMOTIONS
        speakers = [speaker]
        focal_property = 'emotion'
    else:
        raise Exception(f"Failure to render_episode_sentiment_line_chart: freeze_on={freeze_on} is not supported, accepted values are ['emotion', 'speaker']")

    # fetch episode sentiment data and build line chart
    file_path = f'./sentiment_data/{show_key}/openai_emo/{show_key}_{episode_key}.csv'
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        raise Exception(f'Failure to render_episode_sentiment_line_chart: unable to fetch dataframe at file_path={file_path}')
    
    # ick, don't like the second freeze_on check
    if freeze_on == 'emotion':
        print(f'got here 2')
        speakers_series = df['speaker'].value_counts().sort_values(ascending=False)
        speakers = [s for s,_ in speakers_series.items()]        
    
    sentiment_line_chart = fb.build_episode_sentiment_line_chart(show_key, df, speakers, emotions, focal_property)

    return sentiment_line_chart


############ speaker-3d-network-graph callbacks
@dapp_new.callback(
    Output('speaker-3d-network-graph-new', 'figure'),
    Input('show-key3', 'value'),
    Input('episode-key3', 'value'))    
def render_speaker_3d_network_graph_new(show_key: str, episode_key: str):
    print(f'in render_speaker_3d_network_graph_new, show_key={show_key} episode_key={episode_key}')

    # form-backing data
    # episodes = esr.fetch_simple_episodes(ShowKey(show_key))

    # generate data and build generate 3d network graph
    data = esr.speaker_relations_graph(ShowKey(show_key), episode_key)
    fig_scatter = fb.build_3d_network_graph(show_key, data)

    return fig_scatter


############ speaker-frequency-bar-chart callbacks
@dapp_new.callback(
    Output('speaker-episode-frequency-bar-chart-new', 'figure'),
    Input('show-key4', 'value'),
    Input('episode-key4', 'value'),
    Input('span-granularity', 'value'))    
def render_speaker_frequency_bar_chart_new(show_key: str, episode_key: str, span_granularity: str):
    print(f'in render_speaker_frequency_bar_chart_new, show_key={show_key} episode_key={episode_key} span_granularity={span_granularity}')

    # fetch or generate aggregate speaker data and build speaker frequency bar chart
    file_path = f'./app/data/speaker_episode_aggs_{show_key}.csv'
    print(f'loading dataframe at file_path={file_path}')
    df = pd.read_csv(file_path)
    df = df.loc[df['episode_key'] == int(episode_key)]
    
    speaker_episode_frequency_bar_chart = fb.build_speaker_episode_frequency_bar(show_key, episode_key, df, span_granularity)

    return speaker_episode_frequency_bar_chart


if __name__ == "__main__":
    dapp_new.run_server(debug=True)
