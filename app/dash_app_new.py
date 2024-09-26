import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output, State
import os
import pandas as pd
import urllib.parse

import app.dash_new.components as cmp
from app.dash_new import episode_palette, oopsy
import app.es.es_read_router as esr
from app.nlp.nlp_metadata import OPENAI_EMOTIONS
from app.show_metadata import ShowKey, TOPIC_COLORS
import app.fig_builder.fig_helper as fh
import app.fig_builder.plotly_bar as pbar
import app.fig_builder.plotly_gantt as pgantt
import app.fig_builder.plotly_line as pline
import app.fig_builder.plotly_networkgraph as pgraph
import app.fig_builder.plotly_scatter as pscat
import app.fig_builder.plotly_treemap as ptree


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
        # parse show_key from params
        if 'show_key' in parsed_dict:
            show_key = parsed_dict['show_key']
            if isinstance(show_key, list):
                show_key = show_key[0]
        else:
            err_msg = f'show_key is required'
            print(err_msg)
            return oopsy.generate_content(err_msg)

        # all episodes
        all_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
        all_episodes = all_episodes_response['episodes']
        episode_dropdown_options = []
        for episode in all_episodes:
            label = f"{episode['title']} (S{episode['season']}:E{episode['sequence_in_season']})"
            episode_dropdown_options.append({'label': label, 'value': episode['episode_key']})

        # parse episode_key from params
        if 'episode_key' in parsed_dict:
            episode_key = parsed_dict['episode_key']
            if isinstance(episode_key, list):
                episode_key = episode_key[0]

        episode = None
        for ep in all_episodes:
            if ep['episode_key'] == episode_key:
                episode = ep
                break
        if not episode:
            err_msg = f'no episode matching episode_key={episode_key}'
            print(err_msg)
            return oopsy.generate_content(err_msg)

        # supplement episode data with line_count and word_count
        scene_events_by_speaker_response = esr.agg_scene_events_by_speaker(ShowKey(show_key), episode_key=episode_key)
        episode['line_count'] = scene_events_by_speaker_response['scene_events_by_speaker']['_ALL_']

        dialog_word_counts_response = esr.agg_dialog_word_counts(ShowKey(show_key), episode_key=episode_key)
        episode_word_counts = dialog_word_counts_response['dialog_word_counts']
        episode['word_count'] = round(episode_word_counts['_ALL_'])

        # speakers in episode
        speaker_episodes_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key)
        episode_speakers = speaker_episodes_response['speaker_episodes']
        speaker_dropdown_options = ['ALL'] + [s['speaker'] for s in episode_speakers]

        # emotions
        emotion_dropdown_options = ['ALL'] + OPENAI_EMOTIONS

        return episode_palette.generate_content(episode_dropdown_options, episode, speaker_dropdown_options, emotion_dropdown_options)
    


############ episode-palette callbacks
@dapp_new.callback(
    Output('episode-dialog-timeline-new', 'figure'),
    Output('episode-location-timeline-new', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('show-layers', 'value'))    
def render_episode_chromatographs(show_key: str, episode_key: str, show_layers: list):
    print(f'in render_episode_chromatographs, show_key={show_key} episode_key={episode_key}')

    # generate timeline data
    response = esr.generate_episode_gantt_sequence(ShowKey(show_key), episode_key)

    interval_data = None
    if 'scene_locations' in show_layers:
        interval_data = response['location_timeline']

    # build episode gantt charts
    episode_dialog_timeline = pgantt.build_episode_gantt(show_key, 'speakers', response['dialog_timeline'], interval_data=interval_data)
    episode_location_timeline = pgantt.build_episode_gantt(show_key, 'locations', response['location_timeline'])

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
    # freeze_on emotion -> constrain display to one emotion, display all speakers or--for now--selected speaker
    if freeze_on == 'emotion':
        emotions = [emotion]
        if speaker == 'ALL':
            speakers = episode_speakers
        else:
            speakers = [speaker]
        focal_property = 'speaker'
    # freeze_on speaker -> constrain display to one speaker, display all emotions or--for now--selected emotion
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

    return sentiment_line_chart


############ speaker-3d-network-graph callbacks
@dapp_new.callback(
    Output('speaker-3d-network-graph-new', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('scale-by', 'value'))    
def render_speaker_3d_network_graph_new(show_key: str, episode_key: str, scale_by: str):
    print(f'in render_speaker_3d_network_graph_new, show_key={show_key} episode_key={episode_key} scale_by={scale_by}')

    # generate speaker relations data and build 3d network graph
    speaker_relations_data = esr.speaker_relations_graph(ShowKey(show_key), episode_key)

    # NOTE where and how to layer in color mapping is a WIP
    speakers = [n['speaker'] for n in speaker_relations_data['nodes']]
    speaker_colors = fh.generate_speaker_color_discrete_map(show_key, speakers)
    for n in speaker_relations_data['nodes']:
        n['color'] = speaker_colors[n['speaker']].lower() # ugh with the lowercase

    dims = {'height': 800, 'node_max': 60, 'node_min': 12}

    fig_scatter = pgraph.build_speaker_chatter_scatter3d(show_key, speaker_relations_data, scale_by, dims=dims)

    return fig_scatter


############ speaker-frequency-bar-chart callbacks
@dapp_new.callback(
    Output('speaker-episode-frequency-bar-chart-new', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('scale-by', 'value'))    
def render_speaker_frequency_bar_chart_new(show_key: str, episode_key: str, scale_by: str):
    print(f'in render_speaker_frequency_bar_chart_new, show_key={show_key} episode_key={episode_key} scale_by={scale_by}')

    speakers_for_episode_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key)
    speakers_for_episode = speakers_for_episode_response['speaker_episodes']
    df = pd.DataFrame(speakers_for_episode, columns = ['speaker', 'agg_score', 'scene_count', 'line_count', 'word_count'])
    
    speaker_episode_frequency_bar_chart = pbar.build_speaker_episode_frequency_bar(show_key, df, scale_by)

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
    df = pd.DataFrame(speakers_for_episode, columns=['speaker', 'agg_score', 'scene_count', 'line_count', 'word_count'])
    
    speaker_chatter_scatter = pscat.build_speaker_chatter_scatter(df, x_axis, y_axis)

    return speaker_chatter_scatter


############ episode-speaker-topic-scatter callbacks
@dapp_new.callback(
    Output('episode-speaker-mbti-scatter', 'figure'),
    Output('episode-speaker-dnda-scatter', 'figure'),
    Output('episode-speaker-mbti-dt', 'children'),
    Output('episode-speaker-dnda-dt', 'children'),
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

    speaker_color_map = fh.generate_speaker_color_discrete_map(show_key, episode_speaker_names)

    # merge episode-level and series-level speaker topic data (mbti, dnda) for each episode speaker, keeping only the top topic from each context
    flat_speakers = fh.flatten_episode_speaker_topics(episode_speakers, indexed_speakers_response['speakers'])
    
    df = pd.DataFrame(flat_speakers)

    episode_speaker_mbti_scatter = pscat.build_episode_speaker_topic_scatter(show_key, df, 'mbti', speaker_color_map=speaker_color_map)
    episode_speaker_dnda_scatter = pscat.build_episode_speaker_topic_scatter(show_key, df, 'dnda', speaker_color_map=speaker_color_map)

    # build dash datatable
    mbti_display_cols = ['speaker', 'mbti_topic_key', 'mbti_topic_name', 'mbti_score', 'mbti_raw_score']
    episode_speaker_mbti_dt = cmp.pandas_df_to_dash_dt(df, mbti_display_cols, 'speaker', episode_speaker_names, speaker_color_map)
    dnda_display_cols = ['speaker', 'dnda_topic_key', 'dnda_topic_name', 'dnda_score', 'dnda_raw_score']
    episode_speaker_dnda_dt = cmp.pandas_df_to_dash_dt(df, dnda_display_cols, 'speaker', episode_speaker_names, speaker_color_map)

    return episode_speaker_mbti_scatter, episode_speaker_dnda_scatter, episode_speaker_mbti_dt, episode_speaker_dnda_dt


############ episode-topic-treemap callbacks
@dapp_new.callback(
    Output('episode-universal-genres-treemap', 'figure'),
    Output('episode-universal-genres-dt', 'children'),
    Output('episode-universal-genres-gpt35-v2-treemap', 'figure'),
    Output('episode-universal-genres-gpt35-v2-dt', 'children'),
    # Output('episode-focused-gpt35-treemap', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('topic-score-type', 'value'))    
def render_episode_topic_treemap(show_key: str, episode_key: str, topic_score_type: str):
    print(f'in render_episode_topic_treemap, show_key={show_key} episode_key={episode_key}')

    figs = {}
    dts = {}
    # topic_groupings = ['universalGenres', 'universalGenresGpt35_v2', f'focusedGpt35_{show_key}']
    topic_groupings = ['universalGenres', 'universalGenresGpt35_v2']
    for tg in topic_groupings:
        # fetch episode topics, load into df, modify / reformat
        r = esr.fetch_episode_topics(ShowKey(show_key), episode_key, tg)
        episode_topics = r['episode_topics']
        df = pd.DataFrame(episode_topics)
        df = fh.flatten_and_format_topics_df(df, topic_score_type)
        # build treemap fig
        fig = ptree.build_episode_topic_treemap(df.copy(), tg, topic_score_type, max_per_parent=3)
        figs[tg] = fig
        # build dash datatable
        parent_topics = df['parent_topic'].unique()
        display_cols = ['parent_topic', 'topic_name', 'raw_score', 'scaled_score', 'tfidf_score']
        dash_dt = cmp.pandas_df_to_dash_dt(df, display_cols, 'parent_topic', parent_topics, TOPIC_COLORS)
        dts[tg] = dash_dt

    return figs['universalGenres'], dts['universalGenres'], figs['universalGenresGpt35_v2'], dts['universalGenresGpt35_v2']


############ episode-similarity-scatter callbacks
@dapp_new.callback(
    Output('episode-similarity-scatter', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('mlt-type', 'value'))    
def render_episode_similarity_scatter(show_key: str, episode_key: str, mlt_type: str):
    print(f'in render_episode_similarity_scatter, show_key={show_key} episode_key={episode_key} mlt_type={mlt_type}')

    season_response = esr.list_seasons(ShowKey(show_key))
    seasons = season_response['seasons']

    if mlt_type == 'tfidf':
        mlt_response = esr.more_like_this(ShowKey(show_key), episode_key)
        mlt_matches = mlt_response['matches']
        for i, match in enumerate(mlt_matches):
            match['rank'] = i+1
    elif mlt_type == 'openai_embeddings':
        mlt_response = esr.episode_mlt_vector_search(ShowKey(show_key), episode_key)
        mlt_matches = mlt_response['matches'][:30]
        
    high_score = mlt_matches[0]['score']
    simple_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
    all_episodes = simple_episodes_response['episodes']
    all_episodes = [dict(episode, rank=len(mlt_matches)+1, rev_rank=1, score=0, group='all') for episode in all_episodes]
    all_episodes_dict = {episode['episode_key']:episode for episode in all_episodes}
    # transfer rank/score properties from mlt_matches to all_episodes_dict
    for i, mlt_match in enumerate(mlt_matches):
        if mlt_match['episode_key'] in all_episodes_dict:
            ep = all_episodes_dict[mlt_match['episode_key']]
            if mlt_type == 'openai_embeddings':
                ep['rank'] = i+1
            else:
                ep['rank'] = mlt_match['rank']
            ep['rev_rank'] = len(mlt_matches) - ep['rank']
            ep['score'] = mlt_match['score']
            ep['group'] = 'mlt'
    # assign 'highest' rank/score properties to focal episode inside all_episodes_dict
    all_episodes_dict[episode_key]['rank'] = 0
    all_episodes_dict[episode_key]['rev_rank'] = len(mlt_matches) + 1
    all_episodes_dict[episode_key]['score'] = high_score + .01
    all_episodes_dict[episode_key]['group'] = 'focal'

    all_episodes = list(all_episodes_dict.values())

    # load all episodes, with ranks/scores assigned to focal episode and similar episodes, into dataframe
    df = pd.DataFrame(all_episodes)

    # TODO would be great to extract these into a metadata constant like EPISODE_CORE_FIELDS (then add score, rank, & symbol)
    cols_to_keep = ['episode_key', 'title', 'season', 'sequence_in_season', 'air_date', 'score', 'rank', 'rev_rank', 'focal_speakers', 'focal_locations', 
                    'topics_universal', 'topics_focused', 'topics_universal_tfidf', 'topics_focused_tfidf', 'group']

    df = df[cols_to_keep]
    # NOTE sequence matters: sorting this way is an admission of defeat wrt symbol setting
    # px.scatter ignores explicitly set 'diamond' and 'circle' values and goes by df row sequence when assigning traces to symbols
    # Symbol groupings are relevant, but the actual symbol values are ignored (those 2 words could be anything and result would be the same)
    df.sort_values('rev_rank', inplace=True)

    episode_similarity_scatter = pscat.build_episode_similarity_scatter(df, seasons)

    return episode_similarity_scatter


if __name__ == "__main__":
    dapp_new.run_server(debug=True)
