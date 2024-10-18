# import dash
from bertopic import BERTopic
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output, State
import os
import pandas as pd
import urllib.parse

import app.dash.components as cmp
from app.dash import (
    bertopic_model_clusters, episode_gantt_chart, location_line_chart, sentiment_line_chart, series_gantt_chart, series_search_results_gantt, 
    show_3d_network_graph, speaker_3d_network_graph, show_cluster_scatter, show_network_graph, speaker_frequency_bar_chart, speaker_line_chart
)
import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.es.es_read_router as esr
import app.nlp.embeddings_factory as ef
from app.nlp.nlp_metadata import BERTOPIC_DATA_DIR, BERTOPIC_MODELS_DIR, OPENAI_EMOTIONS
from app.show_metadata import ShowKey
import app.utils as utils
import app.data_service.field_meta as fm
import app.fig_builder.plotly_bar as pbar
import app.fig_builder.plotly_bertopic as pbert
import app.fig_builder.plotly_gantt as pgantt
import app.fig_builder.plotly_line as pline
import app.fig_builder.plotly_networkgraph as pgraph
import app.fig_builder.plotly_scatter as pscat
import app.fig_meta.color_meta as cm


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
    
    elif pathname == "/tsp_dash/series-gantt-chart":
        return series_gantt_chart.content
    
    elif pathname == "/tsp_dash/speaker-line-chart":
        return speaker_line_chart.content
    
    elif pathname == "/tsp_dash/location-line-chart":
        return location_line_chart.content
    
    elif pathname == "/tsp_dash/series-search-results-gantt":
        return series_search_results_gantt.content
    
    elif pathname == "/tsp_dash/speaker-frequency-bar-chart":
        return speaker_frequency_bar_chart.content
    
    elif pathname == "/tsp_dash/bertopic-3d-clusters":
        # parse show_key from params and populate bertopic_model_options
        if 'show_key' in parsed_dict:
            show_key = parsed_dict['show_key']
            if isinstance(show_key, list):
                show_key = show_key[0]
        else:
            show_key = 'TNG'
        bertopic_model_list_response = esr.list_bertopic_models(show_key)
        bertopic_model_options = bertopic_model_list_response['bertopic_model_ids']
        # parse bertopic_model_id from params
        bertopic_model_id = None
        if 'bertopic_model_id' in parsed_dict:
            bertopic_model_id = parsed_dict['bertopic_model_id']
            if isinstance(bertopic_model_id, list):
                bertopic_model_id = bertopic_model_id[0]
        return bertopic_model_clusters.generate_content(bertopic_model_options, bertopic_model_id)
    
    elif pathname == "/tsp_dash/sentiment-line-chart":
        # TODO this duplicates speaker-3d-network-graph
        # generate form-backing data 
        # parse episode_key from params
        if 'show_key' in parsed_dict:
            show_key = parsed_dict['show_key']
            if isinstance(show_key, list):
                show_key = show_key[0]
        else:
            show_key = 'TNG'
        # fetch episode listing for dropdown menu
        all_simple_episodes = esr.fetch_simple_episodes(ShowKey(show_key))
        episode_dropdown_options = []
        for episode in all_simple_episodes['episodes']:
            label = f"{episode['title']} (S{episode['season']}:E{episode['sequence_in_season']})"
            episode_dropdown_options.append({'label': label, 'value': episode['episode_key']})
        # parse episode_key from params
        if 'episode_key' in parsed_dict:
            episode_key = parsed_dict['episode_key']
            if isinstance(episode_key, list):
                episode_key = episode_key[0]
        else:
            episode_key = '218'

        speaker_episodes_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key)
        episode_speakers = speaker_episodes_response['speaker_episodes']
        speaker_dropdown_options = ['ALL'] + [s['speaker'] for s in episode_speakers]

        # # load sentiment df to get
        # file_path = f'./sentiment_data/{show_key}/openai_emo/{show_key}_{episode_key}.csv'
        # if os.path.isfile(file_path):
        #     df = pd.read_csv(file_path)
        #     print(f'loading dataframe at file_path={file_path}')
        # else:
        #     raise Exception(f'Failure to render_episode_sentiment_line_chart: unable to fetch dataframe at file_path={file_path}')

        # episode_speakers = df['speaker'].value_counts().sort_values(ascending=False)
        # print(f'episode_speakers={episode_speakers}')
        # print(f'type(episode_speakers)={type(episode_speakers)}')
        # print(f'episode_speakers.iloc[0]={episode_speakers.iloc[0]}')
        # print(f'type(episode_speakers.iloc[0])={type(episode_speakers.iloc[0])}')

        # episode_speaker_options = []
        # for speaker, _ in episode_speakers.items():
        #     episode_speaker_options.append({"label": speaker, "value": speaker})
        
        print(f'episode_speaker_options={speaker_dropdown_options}')

        return sentiment_line_chart.generate_content(episode_key, episode_dropdown_options, speaker_dropdown_options)


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
    s = esqb.fetch_series_embeddings(show_key, vector_field)
    doc_embeddings = esrt.return_all_embeddings(s, vector_field)

    # generate and color-stamp clusters for all show episodes 
    doc_embeddings_clusters_df = ef.cluster_docs(doc_embeddings, num_clusters)
    doc_embeddings_clusters_df['cluster_color'] = doc_embeddings_clusters_df['cluster'].apply(lambda x: cm.colors[x % 10])

    # fetch basic title/season data for all show episodes 
    all_episodes = esr.fetch_simple_episodes(ShowKey(show_key))
    episodes_df = pd.DataFrame(all_episodes['episodes'])

    # merge basic episode data into cluster data
    episodes_df['doc_id'] = episodes_df['episode_key'].apply(lambda x: f'{show_key}_{x}')
    episode_embeddings_clusters_df = pd.merge(doc_embeddings_clusters_df, episodes_df, on='doc_id', how='outer')

    # generate dash_table div as part of callback output
    episode_clusters_df = episode_embeddings_clusters_df[fm.episode_keep_cols + fm.cluster_cols].copy()
    episode_clusters_df = cmp.flatten_and_format_cluster_df(show_key, episode_clusters_df)
    dash_dt = cmp.pandas_df_to_dash_dt(episode_clusters_df, num_clusters)

    # generate scatterplot
    fig_scatter = pscat.build_cluster_scatter(episode_embeddings_clusters_df, show_key, num_clusters)

    return fig_scatter, show_key, dash_dt


############ show-network-graph callbacks
@dapp.callback(
    Output('show-network-graph', 'figure'),
    Output('show-key-display2', 'children'),
    Input('show-key', 'value'))    
def render_show_network_graph(show_key: str):
    print(f'in render_show_network_graph, show_key={show_key}')

    # generate network graph
    fig_scatter = pgraph.build_network_graph()

    return fig_scatter, show_key


# TODO this is out of date and probably not going to be used
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
    fig_scatter = pgraph.build_speaker_chatter_scatter3d(show_key, data)

    return fig_scatter, show_key


############ speaker-3d-network-graph callbacks
@dapp.callback(
    Output('speaker-3d-network-graph', 'figure'),
    Output('show-key-display4', 'children'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('scale-by', 'value'))    
def render_speaker_3d_network_graph(show_key: str, episode_key: str, scale_by: str):
    print(f'in render_speaker_3d_network_graph, show_key={show_key} episode_key={episode_key} scale_by={scale_by}')

    # generate data and build generate 3d network graph
    speaker_relations_data = esr.speaker_relations_graph(ShowKey(show_key), episode_key)

    # NOTE where and how to layer in color mapping is a WIP
    speakers = [n['speaker'] for n in speaker_relations_data['nodes']]
    speaker_colors = cm.generate_speaker_color_discrete_map(show_key, speakers)
    for n in speaker_relations_data['nodes']:
        n['color'] = speaker_colors[n['speaker']].lower() # ugh with the lowercase

    dims = {'height': 800, 'width': 1400, 'node_max': 80, 'node_min': 15}

    fig_scatter = pgraph.build_speaker_chatter_scatter3d(show_key, speaker_relations_data, scale_by, dims=dims)

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
    episode_dialog_timeline = pgantt.build_episode_gantt(show_key, 'speakers', response['dialog_timeline'])
    episode_location_timeline = pgantt.build_episode_gantt(show_key, 'locations', response['location_timeline'])

    return episode_dialog_timeline, episode_location_timeline, show_key


############ series-gantt-chart callbacks
@dapp.callback(
    Output('series-speaker-gantt', 'figure'),
    Output('series-location-gantt', 'figure'),
    Output('series-topic-gantt', 'figure'),
    Output('show-key-display6', 'children'),
    Input('show-key', 'value'))    
def render_series_gantt_chart(show_key: str):
    print(f'in render_series_gantt_chart, show_key={show_key}')

    # speaker gantt
    file_path = f'./app/data/speaker_gantt_sequence_{show_key}.csv'
    if os.path.isfile(file_path):
        speaker_gantt_sequence_df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        print(f'no file found at file_path={file_path}, running `/esr/generate_series_speaker_gantt_sequence/{show_key}?overwrite_file=True` to generate')
        esr.generate_series_speaker_gantt_sequence(ShowKey(show_key), overwrite_file=True, limit_cast=True)
        if os.path.isfile(file_path):
            speaker_gantt_sequence_df = pd.read_csv(file_path)
            print(f'loading dataframe at file_path={file_path}')
        else:
            raise Exception('Failure to render_series_gantt_chart: unable to fetch or generate dataframe at file_path={file_path}')
    series_speaker_gantt = pgantt.build_series_gantt(show_key, speaker_gantt_sequence_df, 'speakers')

    # location gantt
    file_path = f'./app/data/location_gantt_sequence_{show_key}.csv'
    if os.path.isfile(file_path):
        location_gantt_sequence_df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        print(f'no file found at file_path={file_path}, running `/esr/generate_series_location_gantt_sequence/{show_key}?overwrite_file=True` to generate')
        esr.generate_series_location_gantt_sequence(ShowKey(show_key), overwrite_file=True)
        if os.path.isfile(file_path):
            location_gantt_sequence_df = pd.read_csv(file_path)
            print(f'loading dataframe at file_path={file_path}')
        else:
            raise Exception('Failure to render_series_gantt_chart: unable to fetch or generate dataframe at file_path={file_path}')
    series_location_gantt = pgantt.build_series_gantt(show_key, location_gantt_sequence_df, 'locations')

    # topic gantt
    topic_grouping = 'universalGenres'
    # topic_grouping = f'focusedGpt35_{show_key}'
    topic_threshold = 20
    model_vendor= 'openai'
    model_version = 'ada002'
    file_path = f'./app/data/topic_gantt_sequence_{show_key}_{topic_grouping}_{model_vendor}_{model_version}.csv'
    if os.path.isfile(file_path):
        topic_gantt_sequence_df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        print(f'no file found at file_path={file_path}, running `/esr/generate_series_topic_gantt_sequence/{show_key}?overwrite_file=True` to generate')
        esr.generate_series_topic_gantt_sequence(ShowKey(show_key), overwrite_file=True, topic_grouping=topic_grouping, topic_threshold=topic_threshold,
                                                 model_vendor=model_vendor, model_version=model_version)
        if os.path.isfile(file_path):
            topic_gantt_sequence_df = pd.read_csv(file_path)
            print(f'loading dataframe at file_path={file_path}')
        else:
            raise Exception('Failure to render_series_gantt_chart: unable to fetch or generate dataframe at file_path={file_path}')
    series_topic_gantt = pgantt.build_series_gantt(show_key, topic_gantt_sequence_df, 'topics')

    return series_speaker_gantt, series_location_gantt, series_topic_gantt, show_key


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
    
    speaker_line_chart = pline.build_speaker_line_chart(show_key, df, span_granularity, aggregate_ratio=aggregate_ratio, season=season)

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
    
    location_line_chart = pline.build_location_line_chart(show_key, df, span_granularity, aggregate_ratio=aggregate_ratio, season=season)

    return location_line_chart, show_key


############ series-search-results-gantt callbacks
@dapp.callback(
    Output('series-search-results-gantt', 'figure'),
    Output('show-key-display9', 'children'),
    Output('qt-display', 'children'),
    Input('show-key', 'value'),
    Input('qt', 'value'),
    # NOTE: I believe `qt-submit`` is a placebo: it's a call to action, but simply exiting the qt field invokes the callback
    Input('qt-submit', 'value'))    
def render_series_search_results_gantt(show_key: str, qt: str, qt_submit: bool = False):
    print(f'in render_series_gantt_chart, show_key={show_key} qt={qt} qt_submit={qt_submit}')

    # execute search query and filter response into series gantt charts

    # TODO fetch from file, but file has to have all speaker data
    # file_path = f'./app/data/speaker_gantt_sequence_{show_key}.csv'
    # if os.path.isfile(file_path):
    #     speaker_gantt_sequence_df = pd.read_csv(file_path)
    #     print(f'loading dataframe at file_path={file_path}')
    # else:
    #     print(f'no file found at file_path={file_path}, running `/esr/generate_series_speaker_gantt_sequence/{show_key}?overwrite_file=True` to generate')
    #     esr.generate_series_speaker_gantt_sequence(ShowKey(show_key), overwrite_file=True)
    #     if os.path.isfile(file_path):
    #         speaker_gantt_sequence_df = pd.read_csv(file_path)
    #         print(f'loading dataframe at file_path={file_path}')
    #     else:
    #         raise Exception('Failure to render_series_gantt_chart: unable to fetch or generate dataframe at file_path={file_path}')
    
    # search_response = esr.search_scene_events(ShowKey(show_key), dialog=qt)
    # series_search_results_gantt = pgantt.build_series_search_results_gantt(show_key, qt, search_response['matches'], speaker_gantt_sequence_df)

    series_gantt_response = esr.generate_series_speaker_gantt_sequence(ShowKey(show_key))
    search_response = esr.search_scene_events(ShowKey(show_key), dialog=qt)
    # if 'matches' not in search_response or len(search_response['matches']) == 0:
    #     print(f"no matches for show_key={show_key} qt=`{qt}` qt_submit=`{qt_submit}`")
    #     return None, show_key, qt
    # print(f"len(search_response['matches'])={len(search_response['matches'])}")
    # print(f"len(series_gantt_response['episode_speakers_sequence'])={len(series_gantt_response['episode_speakers_sequence'])}")
    timeline_df = pd.DataFrame(series_gantt_response['episode_speakers_sequence'])
    series_search_results_gantt = pgantt.build_series_search_results_gantt(show_key, timeline_df, search_response['matches'])

    return series_search_results_gantt, show_key, qt


############ speaker-frequency-bar-chart callbacks
@dapp.callback(
    Output('speaker-season-frequency-bar-chart', 'figure'),
    Output('speaker-episode-frequency-bar-chart', 'figure'),
    Output('show-key-display10', 'children'),
    Input('show-key', 'value'),
    Input('span-granularity', 'value'),
    Input('season', 'value'),
    Input('sequence-in-season', 'value'))    
def render_speaker_frequency_bar_chart(show_key: str, span_granularity: str, season: str, sequence_in_season: str = None):
    print(f'in render_speaker_frequency_bar_chart, show_key={show_key} span_granularity={span_granularity} season={season} sequence_in_season={sequence_in_season}')

    if season in ['0', 0, 'All']:
        season = None
    else:
        season = int(season)

    # fetch or generate aggregate speaker data and build speaker frequency bar chart
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
            raise Exception('Failure to render_speaker_frequency_bar_chart: unable to fetch or generate dataframe at file_path={file_path}')
    
    speaker_season_frequency_bar_chart = pbar.build_speaker_frequency_bar(show_key, df, span_granularity, aggregate_ratio=False, season=season)
    speaker_episode_frequency_bar_chart = pbar.build_speaker_frequency_bar(show_key, df, span_granularity, aggregate_ratio=True, season=season, sequence_in_season=sequence_in_season)

    return speaker_season_frequency_bar_chart, speaker_episode_frequency_bar_chart, show_key


############ bertopic-model-clusters callbacks
@dapp.callback(
    Output('bertopic-model-clusters', 'figure'),
    Output('bertopic-visualize-barchart', 'figure'),
    Output('bertopic-visualize-topics', 'figure'),
    Output('bertopic-visualize-hierarchy', 'figure'),
    Output('show-key-display11', 'children'),
    Output('bertopic-model-id-display', 'children'),
    Output('episode-narratives-per-cluster-df', 'children'),
    Input('show-key', 'value'),
    Input('bertopic-model-id', 'value'))    
def render_bertopic_model_clusters(show_key: str, bertopic_model_id: str):
    print(f'in render_bertopic_model_clusters, show_key={show_key} bertopic_model_id={bertopic_model_id}')

    # load cluster data for bertopic model 
    bertopic_model_docs_df = pd.read_csv(f'{BERTOPIC_DATA_DIR}/{show_key}/{bertopic_model_id}.csv', sep='\t')

    bertopic_model_docs_df['cluster_title_short'] = bertopic_model_docs_df['cluster_title'].apply(utils.truncate)
    bertopic_model_docs_df['cluster'] = bertopic_model_docs_df['cluster_id']
    num_clusters = len(bertopic_model_docs_df['cluster'].unique())

    # generate dash_table div as part of callback output
    bertopic_model_docs_df = bertopic_model_docs_df[['cluster', 'cluster_title_short', 'Probability', 'wc', 'speaker_group', 'episode_key', 
                                                     'title', 'season', 'sequence_in_season', 'air_date', 'scene_count', 'focal_speakers', 'focal_locations',
                                                     'topics_focused_tfidf_list', 'topics_universal_tfidf_list', 'x_coord', 'y_coord', 'z_coord', 'point_size']]
    bertopic_model_docs_df['cluster_color'] = bertopic_model_docs_df['cluster'].apply(lambda x: cm.colors[x % 10])
    bertopic_model_docs_df.drop(['focal_speakers', 'focal_locations'], axis=1, inplace=True) 
    bertopic_model_docs_df = cmp.flatten_and_format_cluster_df(show_key, bertopic_model_docs_df)
    dash_dt = cmp.pandas_df_to_dash_dt(bertopic_model_docs_df, num_clusters)

    # generate 3d scatter
    bertopic_3d_scatter = pgraph.build_bertopic_model_3d_scatter(show_key, bertopic_model_id, bertopic_model_docs_df)

    # generate topic keyword maps and topic graphs
    mmr_bertopic_model = BERTopic.load(f'{BERTOPIC_MODELS_DIR}/{show_key}/{bertopic_model_id}/mmr')
    openai_bertopic_model = BERTopic.load(f'{BERTOPIC_MODELS_DIR}/{show_key}/{bertopic_model_id}/openai')
    bertopic_visualize_barchart = pbert.build_bertopic_visualize_barchart(mmr_bertopic_model)
    bertopic_visualize_topics = pbert.build_bertopic_visualize_topics(openai_bertopic_model)
    bertopic_visualize_hierarchy = pbert.build_bertopic_visualize_hierarchy(openai_bertopic_model)

    return bertopic_3d_scatter, bertopic_visualize_barchart, bertopic_visualize_topics, bertopic_visualize_hierarchy, show_key, bertopic_model_id, dash_dt


############ sentiment-line-chart callbacks
@dapp.callback(
    Output('sentiment-line-chart', 'figure'),
    # Output('episode-speaker-options', 'options'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('freeze-on', 'value'),
    Input('emotion', 'value'),
    Input('speaker', 'value'))    
def render_episode_sentiment_line_chart(show_key: str, episode_key: str, freeze_on: str, emotion: str, speaker: str):
    print(f'in render_episode_sentiment_line_chart, show_key={show_key} episode_key={episode_key} freeze_on={freeze_on} emotion={emotion} speaker={speaker}')

    # fetch episode sentiment data and build line chart
    file_path = f'./sentiment_data/{show_key}/openai_emo/{show_key}_{episode_key}.csv'
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        raise Exception(f'Failure to render_episode_sentiment_line_chart: unable to fetch dataframe at file_path={file_path}')
    
    # extract episode speakers (may or may not be needed)
    episode_speakers_series = df['speaker'].value_counts().sort_values(ascending=False)
    episode_speakers = [s for s,_ in episode_speakers_series.items()]
    if not speaker:
        speaker = 'ALL'
    if not emotion:
        emotion = 'ALL'

    # TODO `freeze_on` toggle should alter `speaker` and `emotion` pulldown values, in the meantime these conditionals cover many of the gaps
    # freeze_on emotion -> constrain display to one emotion, display all speakers
    if freeze_on == 'emotion':
        emotions = [emotion]
        if speaker == 'ALL':
            speakers = episode_speakers
        else:
            speakers = [speaker]
        focal_property = 'speaker'
    # freeze_on speaker -> constrain display to one speaker, display all emotions
    elif freeze_on == 'speaker':
        if emotion == 'ALL':
            emotions = OPENAI_EMOTIONS
        else:
            emotions = [emotion]
        speakers = [speaker]
        focal_property = 'emotion'
    else:
        raise Exception(f"Failure to render_episode_sentiment_line_chart: freeze_on={freeze_on} is not supported, accepted values are ['emotion', 'speaker']")
    
    sentiment_line_chart = pline.build_episode_sentiment_line_chart(show_key, df, speakers, emotions, focal_property)


    # if freeze_on == 'emotion':
    #     emotions = [emotion]
    #     speakers = []
    #     focal_property = 'speaker'
    # elif freeze_on == 'speaker':
    #     emotions = OPENAI_EMOTIONS
    #     speakers = [speaker]
    #     focal_property = 'emotion'
    # else:
    #     raise Exception(f"Failure to render_episode_sentiment_line_chart: freeze_on={freeze_on} is not supported, accepted values are ['emotion', 'speaker']")

    # # fetch episode sentiment data and build line chart
    # file_path = f'./sentiment_data/{show_key}/openai_emo/{show_key}_{episode_key}.csv'
    # if os.path.isfile(file_path):
    #     df = pd.read_csv(file_path)
    #     print(f'loading dataframe at file_path={file_path}')
    # else:
    #     raise Exception(f'Failure to render_episode_sentiment_line_chart: unable to fetch dataframe at file_path={file_path}')
    
    # # ick, don't like the second freeze_on check
    # if freeze_on == 'emotion':
    #     speakers_series = df['speaker'].value_counts().sort_values(ascending=False)
    #     speakers = [s for s,_ in speakers_series.items()]        
    
    # sentiment_line_chart = pline.build_episode_sentiment_line_chart(show_key, df, speakers, emotions, focal_property)

    return sentiment_line_chart
    

if __name__ == "__main__":
    dapp.run_server(debug=True)
