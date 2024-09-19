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

        # all episodes
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

        # speakers in episode
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
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
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
    Input('show-key', 'value'),
    Input('episode-key', 'value'))    
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
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('span-granularity', 'value'))    
def render_speaker_frequency_bar_chart_new(show_key: str, episode_key: str, span_granularity: str):
    print(f'in render_speaker_frequency_bar_chart_new, show_key={show_key} episode_key={episode_key} span_granularity={span_granularity}')

    speakers_for_episode_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key)
    speakers_for_episode = speakers_for_episode_response['speaker_episodes']
    df = pd.DataFrame(speakers_for_episode, columns = ['speaker', 'agg_score', 'scene_count', 'line_count', 'word_count'])
    
    speaker_episode_frequency_bar_chart = fb.build_speaker_episode_frequency_bar(df, span_granularity)

    return speaker_episode_frequency_bar_chart


############ speaker-chatter-scatter callbacks
@dapp_new.callback(
    Output('speaker-chatter-scatter', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'))    
def render_speaker_chatter_scatter(show_key: str, episode_key: str, x_axis: str, y_axis: str):
    print(f'in render_speaker_chatter_scatter, show_key={show_key} episode_key={episode_key} x_axis={x_axis} y_axis={y_axis}')

    speakers_for_episode_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key)
    speakers_for_episode = speakers_for_episode_response['speaker_episodes']
    df = pd.DataFrame(speakers_for_episode, columns = ['speaker', 'agg_score', 'scene_count', 'line_count', 'word_count'])
    # df = pd.DataFrame(speakers_for_episode)
    
    speaker_chatter_scatter = fb.build_speaker_chatter_scatter(df, x_axis, y_axis)

    return speaker_chatter_scatter


############ episode-speaker-topic-scatter callbacks
@dapp_new.callback(
    Output('episode-speaker-mbti-scatter', 'figure'),
    Output('episode-speaker-dnda-scatter', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'))    
def render_episode_speaker_topic_scatter(show_key: str, episode_key: str):
    print(f'in render_episode_speaker_topic_scatter, show_key={show_key} episode_key={episode_key}')

    # fetch episode speakers
    speakers_for_episode_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key, extra_fields='topics_mbti,topics_dnda')
    episode_speakers = speakers_for_episode_response['speaker_episodes']
    
    # fetched series-level indexed version of episode speakers
    episode_speaker_names = [s['speaker'] for s in episode_speakers]
    indexed_speakers_response = esr.fetch_indexed_speakers(ShowKey(show_key), extra_fields='topics_mbti,topics_dnda', speakers=','.join(episode_speaker_names))
    series_speakers = indexed_speakers_response['speakers']
    series_speaker_dicts = {series_s['speaker']:series_s for series_s in series_speakers}

    # merge episode-level and series-level speaker topic data (mbti, dnda) for each episode speaker, keeping only the top topic from each context
    flat_speakers = []
    for s in episode_speakers:
        if s['word_count'] < 20 and s['line_count'] < 3:
            continue
        flat_s = s.copy()
        flat_speakers.append(flat_s)
        # copy high-scoring topic_mbti and topic_dnda for episode
        ep_topic_mbti = s['topics_mbti'][0]
        flat_s['ep_mbti_topic_key'] = ep_topic_mbti['topic_key']
        flat_s['ep_mbti_topic_name'] = ep_topic_mbti['topic_name']
        flat_s['ep_mbti_score'] = ep_topic_mbti['score']
        flat_s['ep_mbti_raw_score'] = ep_topic_mbti['raw_score']
        del flat_s['topics_mbti']
        ep_topic_dnda = s['topics_dnda'][0]
        flat_s['ep_dnda_topic_key'] = ep_topic_dnda['topic_key']
        flat_s['ep_dnda_topic_name'] = ep_topic_dnda['topic_name']
        flat_s['ep_dnda_score'] = ep_topic_dnda['score']
        flat_s['ep_dnda_raw_score'] = ep_topic_dnda['raw_score']
        del flat_s['topics_dnda']
        # copy high-scoring topic_mbti and topic_dnda for series
        if flat_s['speaker'] in series_speaker_dicts:
            series_s = series_speaker_dicts[flat_s['speaker']]
            ser_topic_mbti = series_s['topics_mbti'][0]
            flat_s['ser_mbti_topic_key'] = ser_topic_mbti['topic_key']
            flat_s['ser_mbti_topic_name'] = ser_topic_mbti['topic_name']
            flat_s['ser_mbti_score'] = ser_topic_mbti['score']
            flat_s['ser_mbti_raw_score'] = ser_topic_mbti['raw_score']
            ser_topic_dnda = series_s['topics_dnda'][0]
            flat_s['ser_dnda_topic_key'] = ser_topic_dnda['topic_key']
            flat_s['ser_dnda_topic_name'] = ser_topic_dnda['topic_name']
            flat_s['ser_dnda_score'] = ser_topic_dnda['score']
            flat_s['ser_dnda_raw_score'] = ser_topic_dnda['raw_score']
    
    df = pd.DataFrame(flat_speakers)

    episode_speaker_mbti_scatter = fb.build_episode_speaker_topic_scatter(df, 'mbti')
    episode_speaker_dnda_scatter = fb.build_episode_speaker_topic_scatter(df, 'dnda')

    return episode_speaker_mbti_scatter, episode_speaker_dnda_scatter


############ episode-topic-treemap callbacks
@dapp_new.callback(
    Output('episode-universal-genres-treemap', 'figure'),
    Output('episode-universal-genres-gpt35-v2-treemap', 'figure'),
    # Output('episode-focused-gpt35-treemap', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('topic-score-type', 'value'))    
def render_episode_topic_treemap(show_key: str, episode_key: str, topic_score_type: str):
    print(f'in render_episode_topic_treemap, show_key={show_key} episode_key={episode_key}')

    figs = []
    # topic_groupings = ['universalGenres', 'universalGenresGpt35_v2', f'focusedGpt35_{show_key}']
    topic_groupings = ['universalGenres', 'universalGenresGpt35_v2']
    for tg in topic_groupings:
        r = esr.fetch_episode_topics(ShowKey(show_key), episode_key, tg)
        episode_topics = r['episode_topics']
        df = pd.DataFrame(episode_topics)
        fig = fb.build_episode_topic_treemap(df, tg, topic_score_type)
        figs.append(fig)

    return tuple(figs)


############ episode-topic-treemap callbacks
@dapp_new.callback(
    Output('episode-similarity-scatter', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('mlt-type', 'value'))    
def render_episode_similarity_scatter(show_key: str, episode_key: str, mlt_type: str):
    print(f'in render_episode_similarity_scatter, show_key={show_key} episode_key={episode_key} mlt_type={mlt_type}')

    episode_response = esr.fetch_episode(ShowKey(show_key), episode_key)
    episode = episode_response['es_episode']

    if mlt_type == 'tfidf':
        mlt_response = esr.more_like_this(ShowKey(show_key), episode_key)
        mlt_matches = mlt_response['matches']
        sim_eps_df = pd.DataFrame(mlt_matches)
        sim_eps_df['rank'] = range(0, len(sim_eps_df))
    elif mlt_type == 'openai_embeddings':
        mlt_response = esr.episode_mlt_vector_search(ShowKey(show_key), episode_key)
        mlt_matches = mlt_response['matches'][:30]
        sim_eps_df = pd.DataFrame(mlt_matches)

    sim_eps_df['rev_rank'] = range(len(sim_eps_df), 0, -1)

    episode_similarity_scatter = fb.build_episode_similarity_scatter(episode, sim_eps_df)

    return episode_similarity_scatter


if __name__ == "__main__":
    dapp_new.run_server(debug=True)
