# import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output, State
import os
import pandas as pd
import urllib.parse

import app.dash.components as cmp
from app.dash import (
    episode_gantt_chart, location_line_chart, series_search_results_gantt, show_3d_network_graph, speaker_3d_network_graph, 
    show_cluster_scatter, show_gantt_chart, show_network_graph, speaker_line_chart
)
import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.es.es_read_router as esr
import app.nlp.embeddings_factory as ef
from app.show_metadata import ShowKey
import app.web.fig_builder as fb
import app.web.fig_metadata as fm


dapp = Dash(__name__,
            external_stylesheets=[dbc.themes.SOLAR],
            requests_pathname_prefix='/tsp_dash/')

# app layout
dapp.layout = dbc.Container(fluid=True, children=[
    cmp.url_bar_and_content_div,
])


# Index callbacks
@dapp.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    Input('url', 'search'))
def display_page(pathname, search):
    # parse params
    parsed = urllib.parse.urlparse(search)
    parsed_dict = urllib.parse.parse_qs(parsed.query)
    print(f'parsed_dict={parsed_dict}')

    if pathname == "/tsp_dash/show-cluster-scatter":
        return show_cluster_scatter.content
    
    elif pathname == "/tsp_dash/show-network-graph":
        return show_network_graph.content
    
    elif pathname == "/tsp_dash/show-3d-network-graph":
        return show_3d_network_graph.content
    
    elif pathname == "/tsp_dash/speaker-3d-network-graph":
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
        return speaker_3d_network_graph.generate_content(episode_dropdown_options, episode_key=episode_key)
    
    elif pathname == "/tsp_dash/episode-gantt-chart":
        # TODO this duplicates speaker-3d-network-graph
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
        return episode_gantt_chart.generate_content(episode_dropdown_options, episode_key=episode_key)
    
    elif pathname == "/tsp_dash/show-gantt-chart":
        return show_gantt_chart.content
    
    elif pathname == "/tsp_dash/speaker-line-chart":
        return speaker_line_chart.content
    
    elif pathname == "/tsp_dash/location-line-chart":
        return location_line_chart.content
    
    elif pathname == "/tsp_dash/series-search-results-gantt":
        return series_search_results_gantt.content


############ show-cluster-scatter callbacks
@dapp.callback(
    Output('show-cluster-scatter', 'figure'),
    Output('show-key-display', 'children'),
    Output('episodes-df-table', 'children'),
    Input('show-key', 'value'),
    Input('num-clusters', 'value'))    
def render_show_cluster_scatter(show_key: str, num_clusters: int):
    print(f'in render_show_cluster_scatter, show_key={show_key} num_clusters={num_clusters}')
    num_clusters = int(num_clusters)
    vector_field = 'openai_ada002_embeddings'

    # fetch embeddings for all show episodes 
    s = esqb.fetch_show_embeddings(show_key, vector_field)
    doc_embeddings = esrt.return_all_embeddings(s, vector_field)

    # generate and color-stamp clusters for all show episodes 
    doc_embeddings_clusters_df = ef.cluster_docs(doc_embeddings, num_clusters)
    doc_embeddings_clusters_df['cluster_color'] = doc_embeddings_clusters_df['cluster'].apply(lambda x: fm.colors[x])

    # fetch basic title/season data for all show episodes 
    all_episodes = esr.fetch_simple_episodes(ShowKey(show_key))
    episodes_df = pd.DataFrame(all_episodes['episodes'])

    # merge basic episode data into cluster data
    episodes_df['doc_id'] = episodes_df['episode_key'].apply(lambda x: f'{show_key}_{x}')
    episode_embeddings_clusters_df = pd.merge(doc_embeddings_clusters_df, episodes_df, on='doc_id', how='outer')

    # generate dash_table div as part of callback output
    episode_clusters_df = episode_embeddings_clusters_df[fm.episode_keep_cols + fm.cluster_cols].copy()
    table_div = cmp.merge_and_simplify_df(episode_clusters_df)

    # generate scatterplot
    fig_scatter = fb.build_cluster_scatter(episode_embeddings_clusters_df, show_key, num_clusters)

    return fig_scatter, show_key, table_div


############ show-network-graph callbacks
@dapp.callback(
    Output('show-network-graph', 'figure'),
    Output('show-key-display2', 'children'),
    Input('show-key', 'value'))    
def render_show_network_graph(show_key: str):
    print(f'in render_show_network_graph, show_key={show_key}')

    # generate network graph
    fig_scatter = fb.build_network_graph()

    return fig_scatter, show_key


############ show-3d-network-graph callbacks
@dapp.callback(
    Output('show-3d-network-graph', 'figure'),
    Output('show-key-display3', 'children'),
    Input('show-key', 'value'))    
def render_show_3d_network_graph(show_key: str):
    print(f'in render_show_3d_network_graph, show_key={show_key}')

    model_vendor = 'es'
    model_version = 'mlt'
    max_edges = 3
    # generate data and build 3d network graph
    data = esr.episode_relations_graph(ShowKey(show_key), model_vendor, model_version, max_edges=max_edges)
    fig_scatter = fb.build_3d_network_graph(show_key, data)

    return fig_scatter, show_key


############ speaker-3d-network-graph callbacks
@dapp.callback(
    Output('speaker-3d-network-graph', 'figure'),
    Output('show-key-display4', 'children'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'))    
def render_speaker_3d_network_graph(show_key: str, episode_key: str):
    print(f'in render_speaker_3d_network_graph, show_key={show_key} episode_key={episode_key}')

    # form-backing data
    # episodes = esr.fetch_simple_episodes(ShowKey(show_key))

    # generate data and build generate 3d network graph
    data = esr.speaker_relations_graph(ShowKey(show_key), episode_key)
    fig_scatter = fb.build_3d_network_graph(show_key, data)

    return fig_scatter, show_key


############ episode-gantt-chart callbacks
@dapp.callback(
    Output('episode-dialog-timeline', 'figure'),
    Output('episode-location-timeline', 'figure'),
    Output('show-key-display5', 'children'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'))    
def render_episode_gantt_chart(show_key: str, episode_key: str):
    print(f'in render_episode_gantt_chart, show_key={show_key} episode_key={episode_key}')

    # generate data and build episode gantt charts
    response = esr.generate_episode_gantt_sequence(ShowKey(show_key), episode_key)
    episode_dialog_timeline = fb.build_episode_gantt(show_key, response['dialog_timeline'])
    episode_location_timeline = fb.build_episode_gantt(show_key, response['location_timeline'])

    return episode_dialog_timeline, episode_location_timeline, show_key


############ show-gantt-chart callbacks
@dapp.callback(
    Output('show-speaker-gantt', 'figure'),
    Output('show-location-gantt', 'figure'),
    Output('show-key-display6', 'children'),
    Input('show-key', 'value'))    
def render_series_gantt_chart(show_key: str):
    print(f'in render_series_gantt_chart, show_key={show_key}')

    # generate data and build series gantt charts
    response = esr.generate_series_gantt_sequence(ShowKey(show_key))
    series_speaker_gantt = fb.build_series_gantt(show_key, response['episode_speakers_sequence'], 'speakers')
    series_location_gantt = fb.build_series_gantt(show_key, response['episode_locations_sequence'], 'locations')

    return series_speaker_gantt, series_location_gantt, show_key


############ speaker-line-chart callbacks
@dapp.callback(
    Output('speaker-line-chart', 'figure'),
    Output('show-key-display7', 'children'),
    Input('show-key', 'value'),
    Input('span-granularity', 'value'),
    Input('aggregate-ratio', 'value'),
    Input('season', 'value'))    
def render_series_speaker_line_chart(show_key: str, span_granularity: str, aggregate_ratio: str, season: str):
    print(f'in render_series_speaker_line_chart, show_key={show_key} span_granularity={span_granularity} aggregate_ratio={aggregate_ratio} season={season}')

    if aggregate_ratio == 'True':
        aggregate_ratio = True
    else:
        aggregate_ratio = False
    if season == 'All':
        season = None
    else:
        season = int(season)

    # fetch or generate aggregate speaker data and build speaker line chart
    # response = esr.generate_speaker_line_chart_sequence(ShowKey(show_key), span_granularity=span_granularity, aggregate_ratio=aggregate_ratio, season=season)
    file_path = f'./app/data/speaker_episode_aggs_{show_key}.csv'
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        print(f'no file found at file_path={file_path}, running `/esr/generate_speaker_line_chart_sequences/{show_key}?overwrite_file=True` to generate')
        esr.generate_speaker_line_chart_sequences(ShowKey(show_key), overwrite_file=True)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            print(f'loading dataframe at file_path={file_path}')
        else:
            raise Exception('Failure to render_series_speaker_line_chart: unable to fetch or generate dataframe at file_path={file_path}')
    
    speaker_line_chart = fb.build_speaker_line_chart(show_key, df, span_granularity, aggregate_ratio=aggregate_ratio, season=season)

    return speaker_line_chart, show_key


############ location-line-chart callbacks
@dapp.callback(
    Output('location-line-chart', 'figure'),
    Output('show-key-display8', 'children'),
    Input('show-key', 'value'),
    Input('span-granularity', 'value'),
    Input('aggregate-ratio', 'value'),
    Input('season', 'value'))    
def render_series_location_line_chart(show_key: str, span_granularity: str, aggregate_ratio: str, season: str):
    print(f'in render_series_speaker_line_chart, show_key={show_key} span_granularity={span_granularity} aggregate_ratio={aggregate_ratio} season={season}')

    if aggregate_ratio == 'True':
        aggregate_ratio = True
    else:
        aggregate_ratio = False
    if season == 'All':
        season = None
    else:
        season = int(season)

    # fetch or generate aggregate location data and build location line chart
    file_path = f'./app/data/location_episode_aggs_{show_key}.csv'
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        print(f'no file found at file_path={file_path}, running `/esr/generate_location_line_chart_sequences/{show_key}?overwrite_file=True` to generate')
        esr.generate_location_line_chart_sequences(ShowKey(show_key), overwrite_file=True)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            print(f'loading dataframe at file_path={file_path}')
        else:
            raise Exception('Failure to render_series_location_line_chart: unable to fetch or generate dataframe at file_path={file_path}')
    
    location_line_chart = fb.build_location_line_chart(show_key, df, span_granularity, aggregate_ratio=aggregate_ratio, season=season)

    return location_line_chart, show_key


############ series-search-results-gantt callbacks
@dapp.callback(
    Output('series-search-results-gantt', 'figure'),
    Output('show-key-display9', 'children'),
    Output('qt-display', 'children'),
    Input('show-key', 'value'),
    Input('qt', 'value'))
    # Input('qt-submit', 'value'))    
def render_series_search_results_gantt(show_key: str, qt: str):
    print(f'in render_series_gantt_chart, show_key={show_key} qt={qt}')

    # execute search query and filter response into series gantt charts
    series_gantt_response = esr.generate_series_gantt_sequence(ShowKey(show_key))
    search_response = esr.search_scene_events(ShowKey(show_key), dialog=qt)
    series_search_results_gantt = fb.series_search_results_gantt(show_key, qt, search_response['matches'], series_gantt_response['episode_speakers_sequence'])

    return series_search_results_gantt, show_key, qt


if __name__ == "__main__":
    dapp.run_server(debug=True)
