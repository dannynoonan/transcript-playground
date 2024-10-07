import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output, State
from operator import itemgetter
import os
import pandas as pd
import urllib.parse

import app.dash_new.components as cmp
from app.dash_new import episode_palette, series_palette, oopsy
import app.es.es_read_router as esr
from app.nlp.nlp_metadata import OPENAI_EMOTIONS
from app.show_metadata import ShowKey, TOPIC_COLORS, show_metadata
import app.fig_builder.fig_helper as fh
import app.fig_builder.plotly_bar as pbar
import app.fig_builder.plotly_gantt as pgantt
import app.fig_builder.plotly_line as pline
import app.fig_builder.plotly_networkgraph as pgraph
import app.fig_builder.plotly_pie as ppie
import app.fig_builder.plotly_scatter as pscat
import app.fig_builder.plotly_treemap as ptree
from app import utils


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
    utils.hilite_in_logs(f'NEW PAGE LOAD with parsed_dict={parsed_dict}')

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
        
        # parse episode_key from params
        if 'episode_key' in parsed_dict:
            episode_key = parsed_dict['episode_key']
            if isinstance(episode_key, list):
                episode_key = episode_key[0]
        else:
            err_msg = f'episode_key is required'
            print(err_msg)
            return oopsy.generate_content(err_msg)
        
        # all seasons
        all_seasons_response = esr.list_seasons(ShowKey(show_key))
        all_seasons = all_seasons_response['seasons']

        # all episodes
        all_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
        all_episodes = all_episodes_response['episodes']
        episode_dropdown_options = []
        for episode in all_episodes:
            label = f"{episode['title']} (S{episode['season']}:E{episode['sequence_in_season']})"
            episode_dropdown_options.append({'label': label, 'value': episode['episode_key']})

        # emotions
        emotion_dropdown_options = ['ALL'] + OPENAI_EMOTIONS

        return episode_palette.generate_content(show_key, episode_key, all_seasons, episode_dropdown_options, emotion_dropdown_options)
    
    elif pathname == "/tsp_dash_new/series-palette":
        # parse show_key from params
        if 'show_key' in parsed_dict:
            show_key = parsed_dict['show_key']
            if isinstance(show_key, list):
                show_key = show_key[0]
        else:
            err_msg = f'show_key is required'
            print(err_msg)
            return oopsy.generate_content(err_msg)
        
        # all seasons
        all_seasons_response = esr.list_seasons(ShowKey(show_key))
        all_seasons = all_seasons_response['seasons']

        # # all episodes
        # all_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
        # all_episodes = all_episodes_response['episodes']
        # episode_dropdown_options = []
        # for episode in all_episodes:
        #     label = f"{episode['title']} (S{episode['season']}:E{episode['sequence_in_season']})"
        #     episode_dropdown_options.append({'label': label, 'value': episode['episode_key']})

        # # emotions
        # emotion_dropdown_options = ['ALL'] + OPENAI_EMOTIONS

        universal_genres_parent_topics = []
        topic_grouping_response = esr.fetch_topic_grouping('universalGenres')
        for t in topic_grouping_response['topics']:
            # only process topics that have parents (ignore the parents themselves)
            if not t['parent_key']:
                universal_genres_parent_topics.append(t['topic_key'])

        return series_palette.generate_content(show_key, all_seasons, universal_genres_parent_topics)
    


######################################## EPISODE CALLBACKS ##########################################

############ episode summary callbacks
@dapp_new.callback(
    Output('episode-title-summary', 'children'),
    Output('episode-scene-count', 'children'),
    Output('episode-line-count', 'children'),
    Output('episode-word-count', 'children'),
    Output('episode-focal-speakers', 'children'),
    Output('episode-topics', 'children'),
    Output('episode-speakers', 'options'),
    Output('episode-wordcloud-img', 'src'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'))    
def render_episode_summary(show_key: str, episode_key: str):
    print(f'in render_episode_summary, show_key={show_key} episode_key={episode_key}')

    episode_response = esr.fetch_episode(ShowKey(show_key), episode_key)
    if not 'es_episode' in episode_response:
        err_msg = f'no episode matching show_key={show_key} episode_key={episode_key}'
        print(err_msg)
        # TODO

    episode = episode_response['es_episode']
    title_summary = f"Season {episode['season']}, Episode {episode['sequence_in_season']}: \"{episode['title']}\""
    if 'air_date' in episode:
        title_summary = f"{title_summary} ({episode['air_date'][:10]})"
    scene_count = episode['scene_count']
    if 'focal_speakers' in episode:
        focal_speakers = ', '.join(episode['focal_speakers'])
    else:
        focal_speakers = ''
    if 'topics_universal_tfidf' in episode:
        parent_topics = [t['topic_key'].split('.')[0] for t in episode['topics_universal_tfidf']]
        distinct_parent_topics = list(dict.fromkeys(parent_topics))
        parent_topics_tfidf = ', '.join(distinct_parent_topics)
    else:
        parent_topics_tfidf = ''
    
    # supplement episode data with line_count and word_count
    scene_events_by_speaker_response = esr.agg_scene_events_by_speaker(ShowKey(show_key), episode_key=episode_key)
    if 'scene_events_by_speaker' in scene_events_by_speaker_response and '_ALL_' in scene_events_by_speaker_response['scene_events_by_speaker']:
        line_count = scene_events_by_speaker_response['scene_events_by_speaker']['_ALL_']
    else:
        line_count = ''

    dialog_word_counts_response = esr.agg_dialog_word_counts(ShowKey(show_key), episode_key=episode_key)
    if 'dialog_word_counts' in dialog_word_counts_response and '_ALL_' in dialog_word_counts_response['dialog_word_counts']:
        word_count = round(dialog_word_counts_response['dialog_word_counts']['_ALL_'])
    else:
        word_count = ''

    # speakers in episode, for sentiment timeline pulldown
    speaker_episodes_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key)
    speaker_episodes = speaker_episodes_response['speaker_episodes']
    episode_speakers = ['ALL'] + [s['speaker'] for s in speaker_episodes]

    wordcloud_img = f"/static/wordclouds/{show_key}/{show_key}_{episode_key}.png"

    return title_summary, scene_count, line_count, word_count, focal_speakers, parent_topics_tfidf, episode_speakers, wordcloud_img


############ episode gantt callbacks
@dapp_new.callback(
    Output('episode-dialog-timeline-new', 'figure'),
    Output('episode-location-timeline-new', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('show-layers', 'value'))    
def render_episode_gantts(show_key: str, episode_key: str, show_layers: list):
    print(f'in render_episode_gantts, show_key={show_key} episode_key={episode_key}')

    # generate timeline data
    response = esr.generate_episode_gantt_sequence(ShowKey(show_key), episode_key)

    interval_data = None
    if 'scene_locations' in show_layers:
        interval_data = response['location_timeline']

    # build episode gantt charts
    episode_dialog_timeline = pgantt.build_episode_gantt(show_key, 'speakers', response['dialog_timeline'], interval_data=interval_data)
    episode_location_timeline = pgantt.build_episode_gantt(show_key, 'locations', response['location_timeline'])

    return episode_dialog_timeline, episode_location_timeline


############ episode search gantt callbacks
@dapp_new.callback(
    Output('out-text', 'children'),
    Output('episode-search-results-gantt', 'figure'),
    Output('episode-search-results-dt', 'children'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('qt', 'value'))    
def render_episode_search_gantt(show_key: str, episode_key: str, qt: str):
    print(f'in render_episode_search_gantt, show_key={show_key} episode_key={episode_key} qt={qt}')

    scene_count = None
    scene_event_count = None

    if not qt:
        return 'No query specified', {}, ''
    
    match_coords = []
    search_response = esr.search_scene_events(ShowKey(show_key), episode_key=str(episode_key), dialog=qt)
    scene_count = search_response['scene_count']
    scene_event_count = search_response['scene_event_count']
    if scene_count > 0:
        episode = search_response['matches'][0]
        for scene in episode['scenes']:
            for scene_event in scene['scene_events']:
                match_coords.append((scene['sequence'], scene_event['sequence'], scene_event['dialog']))

    if not match_coords:
        return f"No episode dialog matching query '{qt}'", {}, ''

    # generate timeline data
    response = esr.generate_episode_gantt_sequence(ShowKey(show_key), episode_key)

    # load full time-series sequence of speakers by episode into a dataframe
    df = pd.DataFrame(response['dialog_timeline'])

    # not all scenes have dialog, and index position in `location_timeline` can't be trusted, so load scene-index-to-scene-location dict
    scene_i_to_locations = {scene['scene']:scene['Task'] for scene in response['location_timeline']}

    # for df rows corresponding to scene_event_coords in df: (1) set highlight col to yes, and (2) replace Line with marked-up search result dialog
    df['highlight'] = 'no'
    df['location'] = 'NA'
    for scene_event_coords in match_coords:
        df.loc[(df['scene'] == scene_event_coords[0]) & (df['scene_event'] == scene_event_coords[1]), 'highlight'] = 'yes'
        df.loc[(df['scene'] == scene_event_coords[0]) & (df['scene_event'] == scene_event_coords[1]), 'Line'] = scene_event_coords[2]
        df.loc[(df['scene'] == scene_event_coords[0]) & (df['scene_event'] == scene_event_coords[1]), 'location'] = scene_i_to_locations[scene_event_coords[0]]
    matching_lines_df = df[df['highlight'] == 'yes']

    # build gantt chart
    episode_search_results_gantt = pgantt.build_episode_search_results_gantt(show_key, df, matching_lines_df)

    # build dash datatable
    matching_lines_df.rename(columns={'Task': 'character', 'scene_event': 'line', 'Line': 'dialog'}, inplace=True)
    matching_speakers = list(matching_lines_df['character'].unique())
    speaker_color_map = fh.generate_speaker_color_discrete_map(show_key, matching_speakers)
    # TODO matching_lines_df['dialog'] = matching_lines_df['dialog'].apply(convert_markup)
    display_cols = ['character', 'scene', 'line', 'location', 'dialog']
    episode_search_results_dt = cmp.pandas_df_to_dash_dt(matching_lines_df, display_cols, 'character', matching_speakers, speaker_color_map, 
                                                         numeric_precision_overrides={'scene': 0, 'line': 0})

    out_text = f"{scene_event_count} lines matching query '{qt}'"

    return out_text, episode_search_results_gantt, episode_search_results_dt


############ sentiment line chart callbacks
@dapp_new.callback(
    Output('sentiment-line-chart-new', 'figure'),
    # Output('episode-speaker-options', 'options'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('freeze-on', 'value'),
    Input('emotion', 'value'),
    Input('episode-speakers', 'value'))    
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


############ speaker 3d network graph callbacks
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

    # TODO gross
    if scale_by == 'scenes':
        scale_by = 'scene_count'
    elif scale_by == 'lines':
        scale_by = 'line_count'
    elif scale_by == 'words':
        scale_by = 'word_count'

    fig_scatter = pgraph.build_speaker_chatter_scatter3d(show_key, speaker_relations_data, scale_by, dims=dims)

    return fig_scatter


############ speaker frequency bar chart callbacks
@dapp_new.callback(
    Output('speaker-episode-frequency-bar-chart-new', 'figure'),
    Output('speaker-episode-summary-dt', 'children'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('scale-by', 'value'))    
def render_speaker_frequency_bar_chart_new(show_key: str, episode_key: str, scale_by: str):
    print(f'in render_speaker_frequency_bar_chart_new, show_key={show_key} episode_key={episode_key} scale_by={scale_by}')

    speakers_for_episode_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key, extra_fields='topics_mbti')
    speakers_for_episode = speakers_for_episode_response['speaker_episodes']
    speakers_for_episode = fh.flatten_speaker_topics(speakers_for_episode, 'mbti', 3)

    df = pd.DataFrame(speakers_for_episode, columns=['speaker', 'agg_score', 'scene_count', 'line_count', 'word_count', 'topics_mbti'])

    episode_speaker_names = [s['speaker'] for s in speakers_for_episode]
    speaker_color_map = fh.generate_speaker_color_discrete_map(show_key, episode_speaker_names)

    # TODO incorporate episode-level sentiment into es writer workflow; for now it's a quick lookup in episode-level dfs
    emo_limit = 3
    file_path = f'sentiment_data/{show_key}/openai_emo/{show_key}_{episode_key}.csv'
    if not os.path.isfile(file_path):
        utils.hilite_in_logs(f'No sentiment data found at file_path={file_path}, continuing without it')
        pass
    else:
        df['emotions'] = ''
        ep_df = pd.read_csv(file_path)

        # NOTE bizarre: converting 'score' to float only needed when callback component invoked by dropdown menu selection, not by full page refresh
        ep_df['score'].apply(lambda x: float(x))

        emotions = list(ep_df['emotion'].unique())
        for spkr in episode_speaker_names:
            spkr_df = ep_df[ep_df['speaker'] == spkr]
            spkr_emo_avgs = []
            for emo in emotions:
                emo_df = spkr_df[spkr_df['emotion'] == emo]
                spkr_emo_avgs.append((emo, round(emo_df['score'].mean(), 4)))
            spkr_emo_avgs.sort(key=itemgetter(1), reverse=True)
            spkr_emo_avgs = spkr_emo_avgs[:emo_limit]
            spkr_emos = [sea[0] for sea in spkr_emo_avgs]
            df.loc[df['speaker'] == spkr, 'emotions'] = ', '.join(spkr_emos)

    df.rename(columns={'speaker': 'character', 'scene_count': 'scenes', 'line_count': 'lines', 'word_count': 'words'}, inplace=True)

    speaker_episode_frequency_bar_chart = pbar.build_speaker_episode_frequency_bar(show_key, df, scale_by)

    speaker_episode_summary_dt = cmp.pandas_df_to_dash_dt(df, df.columns, 'character', episode_speaker_names, speaker_color_map, 
                                                          numeric_precision_overrides={'scenes': 0, 'lines': 0, 'words': 0})
        
    return speaker_episode_frequency_bar_chart, speaker_episode_summary_dt


############ episode similarity scatter callbacks
@dapp_new.callback(
    Output('episode-similarity-scatter', 'figure'),
    Output('episode-similarity-dt', 'children'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('mlt-type', 'value'),
    Input('show-similar-episodes-dt', 'value'))    
def render_episode_similarity_scatter(show_key: str, episode_key: str, mlt_type: str, show_dt: list):
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

    if 'yes' in show_dt:
        # NOTE last-minute first draft effort to sync datatable colors with matplotlib/plotly figure color gradient
        display_cols = ['title', 'season', 'episode', 'score', 'rank', 'flattened_topics']
        df = df.loc[df['score'] > 0]
        df = df.loc[df['rank'] > 0]
        df.sort_values('rank', inplace=True, ascending=True)
        similar_episode_scores = list(df['score'].values)
        viridis_discrete_rgbs = fh.matplotlib_gradient_to_rgb_strings('viridis')
        sim_ep_rgbs = fh.map_range_values_to_gradient(similar_episode_scores, viridis_discrete_rgbs)
        # sim_ep_rgb_textcolors = {rgb:"Black" for rgb in sim_ep_rgbs}
        episode_similarity_dt = cmp.pandas_df_to_dash_dt(df, display_cols, 'rank', sim_ep_rgbs, {}, numeric_precision_overrides={'season': 0, 'episode': 0, 'rank': 0})
    else: 
        episode_similarity_dt = {}

    return episode_similarity_scatter, episode_similarity_dt


############ episode speaker topic grid callbacks
@dapp_new.callback(
    Output('episode-speaker-mbti-scatter', 'figure'),
    Output('episode-speaker-dnda-scatter', 'figure'),
    Output('episode-speaker-mbti-dt', 'children'),
    Output('episode-speaker-dnda-dt', 'children'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('episode-mbti-count', 'value'),
    Input('episode-dnda-count', 'value'))    
def render_episode_speaker_topic_scatter(show_key: str, episode_key: str, mbti_count: int, dnda_count: int):
    print(f'in render_episode_speaker_topic_scatter, show_key={show_key} episode_key={episode_key} mbti_count={mbti_count} dnda_count={dnda_count}')

    # fetch episode speakers
    speakers_for_episode_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key, extra_fields='topics_mbti,topics_dnda')
    episode_speakers = speakers_for_episode_response['speaker_episodes']
    
    # fetched series-level indexed version of episode speakers
    episode_speaker_names = [s['speaker'] for s in episode_speakers]
    # indexed_speakers_response = esr.fetch_indexed_speakers(ShowKey(show_key), extra_fields='topics_mbti,topics_dnda', speakers=','.join(episode_speaker_names))

    speaker_color_map = fh.generate_speaker_color_discrete_map(show_key, episode_speaker_names)

    # NOTE ended up not using this data downstream
    # mbti_distribution_response = esr.agg_numeric_distrib_into_percentiles(ShowKey(show_key), 'speaker_episode_topics', 'raw_score', constraints='topic_grouping:meyersBriggsKiersey')
    # dnda_distribution_response = esr.agg_numeric_distrib_into_percentiles(ShowKey(show_key), 'speaker_episode_topics', 'raw_score', constraints='topic_grouping:dndAlignments')

    # mbti_percent_distrib = mbti_distribution_response["percentile_distribution"]
    # mbti_percent_distrib_list = list(mbti_percent_distrib.values())
    # dnda_percent_distrib = dnda_distribution_response["percentile_distribution"]
    # dnda_percent_distrib_list = list(dnda_percent_distrib.values())

    # flatten episode speaker topic data for each episode speaker
    exploded_speakers_mbti = fh.explode_speaker_topics(episode_speakers, 'mbti', limit_per_speaker=mbti_count)
    exploded_speakers_dnda = fh.explode_speaker_topics(episode_speakers, 'dnda', limit_per_speaker=dnda_count)
    mbti_df = pd.DataFrame(exploded_speakers_mbti)
    dnda_df = pd.DataFrame(exploded_speakers_dnda)
    episode_speaker_mbti_scatter = pscat.build_speaker_topic_scatter(show_key, mbti_df.copy(), 'mbti', speaker_color_map=speaker_color_map)
    episode_speaker_dnda_scatter = pscat.build_speaker_topic_scatter(show_key, dnda_df.copy(), 'dnda', speaker_color_map=speaker_color_map)

    # build dash datatable
    display_cols = ['speaker', 'topic_key', 'topic_name', 'score', 'raw_score']
    episode_speaker_mbti_dt = cmp.pandas_df_to_dash_dt(mbti_df, display_cols, 'speaker', episode_speaker_names, speaker_color_map)
    episode_speaker_dnda_dt = cmp.pandas_df_to_dash_dt(dnda_df, display_cols, 'speaker', episode_speaker_names, speaker_color_map)

    return episode_speaker_mbti_scatter, episode_speaker_dnda_scatter, episode_speaker_mbti_dt, episode_speaker_dnda_dt


############ episode topic treemap callbacks
@dapp_new.callback(
    Output('episode-universal-genres-treemap', 'figure'),
    Output('episode-universal-genres-dt', 'children'),
    Output('episode-universal-genres-gpt35-v2-treemap', 'figure'),
    Output('episode-universal-genres-gpt35-v2-dt', 'children'),
    # Output('episode-focused-gpt35-treemap', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('universal-genres-score-type', 'value'),
    Input('universal-genres-gpt35-v2-score-type', 'value'))    
def render_episode_topic_treemap(show_key: str, episode_key: str, ug_score_type: str, ug2_score_type: str):
    print(f'in render_episode_topic_treemap, show_key={show_key} episode_key={episode_key} ug_score_type={ug_score_type} ug2_score_type={ug2_score_type}')

    figs = {}
    dts = {}
    # topic_groupings = ['universalGenres', 'universalGenresGpt35_v2', f'focusedGpt35_{show_key}']
    topic_groupings = ['universalGenres', 'universalGenresGpt35_v2']
    topic_score_types = [ug_score_type, ug2_score_type]
    for i, tg in enumerate(topic_groupings):
        # fetch episode topics, load into df, modify / reformat
        r = esr.fetch_episode_topics(ShowKey(show_key), episode_key, tg)
        episode_topics = r['episode_topics']
        df = pd.DataFrame(episode_topics)
        df = fh.flatten_and_format_topics_df(df, topic_score_types[i])
        # build treemap fig
        fig = ptree.build_episode_topic_treemap(df.copy(), tg, topic_score_types[i], max_per_parent=3)
        figs[tg] = fig
        # build dash datatable
        parent_topics = df['parent_topic'].unique()
        display_cols = ['parent_topic', 'topic_name', 'raw_score', 'scaled_score', 'tfidf_score']
        dash_dt = cmp.pandas_df_to_dash_dt(df, display_cols, 'parent_topic', parent_topics, TOPIC_COLORS)
        dts[tg] = dash_dt

    return figs['universalGenres'], dts['universalGenres'], figs['universalGenresGpt35_v2'], dts['universalGenresGpt35_v2']


# # NOTE not being used
# ############ episode speaker chatter scatter callbacks
# @dapp_new.callback(
#     Output('speaker-chatter-scatter', 'figure'),
#     Input('show-key', 'value'),
#     Input('episode-key', 'value'),
#     Input('x-axis', 'value'),
#     Input('y-axis', 'value'))    
# def render_speaker_chatter_scatter(show_key: str, episode_key: str, x_axis: str, y_axis: str):
#     print(f'in render_speaker_chatter_scatter, show_key={show_key} episode_key={episode_key} x_axis={x_axis} y_axis={y_axis}')

#     speakers_for_episode_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key)
#     speakers_for_episode = speakers_for_episode_response['speaker_episodes']
#     df = pd.DataFrame(speakers_for_episode, columns=['speaker', 'agg_score', 'scene_count', 'line_count', 'word_count'])
    
#     speaker_chatter_scatter = pscat.build_speaker_chatter_scatter(df, x_axis, y_axis)

#     return speaker_chatter_scatter



######################################## SERIES CALLBACKS ##########################################

############ series summary callbacks
@dapp_new.callback(
    Output('series-title-summary', 'children'),
    Output('series-season-count', 'children'),
    Output('series-episode-count', 'children'),
    Output('series-scene-count', 'children'),
    Output('series-line-count', 'children'),
    Output('series-word-count', 'children'),
    Output('series-air-date-range', 'children'),
    Output('series-wordcloud-img', 'src'),
    # Output('series-topics', 'children'),
    Input('show-key', 'value'))    
def render_series_summary(show_key: str):
    print(f'in render_series_summary, show_key={show_key}')

    '''
    <h3><strong>{{ tdata['show_key'] }}</strong> show page <small>({{ tdata['air_date_range'] }})</small></h3>
    <h5>{{ tdata['season_count'] }} seasons, {{ tdata['episode_count'] }} episodes, {{ tdata['scene_count'] }} scenes, {{ tdata['line_count'] }} lines, {{ tdata['word_count'] }} words</h5>
    '''
    series_title_summary = 'Star Trek: The Next Generation'

    list_seasons_response = esr.list_seasons(ShowKey(show_key))
    all_seasons = list_seasons_response['seasons']
    season_count = len(all_seasons)

    episode_count = 0

    series_speaker_scene_counts_response = esr.agg_scenes_by_speaker(ShowKey(show_key))
    scene_count = series_speaker_scene_counts_response['scenes_by_speaker']['_ALL_']

    series_speakers_response = esr.agg_scene_events_by_speaker(ShowKey(show_key))
    line_count = series_speakers_response['scene_events_by_speaker']['_ALL_']

    series_speaker_word_counts_response = esr.agg_dialog_word_counts(ShowKey(show_key))
    word_count = int(series_speaker_word_counts_response['dialog_word_counts']['_ALL_'])

    # series_speaker_episode_counts_response = esr.agg_episodes_by_speaker(show_key)
    # speaker_count = series_speaker_episode_counts_response['speaker_count']	

    # series_locations_response = esr.agg_scenes_by_location(show_key)
    # location_count = series_locations_response['location_count']

    episodes_by_season = esr.list_simple_episodes_by_season(ShowKey(show_key))
    episodes_by_season = episodes_by_season['episodes_by_season']

    episode_count = 0
    first_episode_in_series = None
    last_episode_in_series = None
    stats_by_season = {}

    for season in episodes_by_season.keys():
        season_episode_count = len(episodes_by_season[season])
        episode_count += len(episodes_by_season[season])
        season_stats = {}

        scenes_by_location_response = esr.agg_scenes_by_location(ShowKey(show_key), season=season)
        season_stats['location_count'] = scenes_by_location_response['location_count']
        season_stats['location_counts'] = utils.truncate_dict(scenes_by_location_response['scenes_by_location'], season_episode_count, start_index=1)

        scene_events_by_speaker_response = esr.agg_scene_events_by_speaker(ShowKey(show_key), season=season)
        season_stats['line_count'] = scene_events_by_speaker_response['scene_events_by_speaker']['_ALL_']
        season_stats['speaker_line_counts'] = utils.truncate_dict(scene_events_by_speaker_response['scene_events_by_speaker'], season_episode_count, start_index=1)
        
        scenes_by_speaker_response = esr.agg_scenes_by_speaker(ShowKey(show_key), season=season)
        season_stats['scene_count'] = scenes_by_speaker_response['scenes_by_speaker']['_ALL_']

        episodes_by_speaker_resopnse = esr.agg_episodes_by_speaker(ShowKey(show_key), season=season)
        season_stats['speaker_count'] = episodes_by_speaker_resopnse['speaker_count']

        word_counts_response = esr.agg_dialog_word_counts(ShowKey(show_key), season=season)
        season_stats['word_count'] = int(word_counts_response['dialog_word_counts']['_ALL_'])

		# generate air_date_range
        first_episode_in_season = episodes_by_season[season][0]
        last_episode_in_season = episodes_by_season[season][-1]
        season_stats['air_date_range'] = f"{first_episode_in_season['air_date'][:10]} - {last_episode_in_season['air_date'][:10]}"
        if not first_episode_in_series:
            first_episode_in_series = episodes_by_season[season][0]
        last_episode_in_series = episodes_by_season[season][-1]
        stats_by_season[season] = season_stats

    air_date_range = f"{first_episode_in_series['air_date'][:10]} - {last_episode_in_series['air_date'][:10]}"

    wordcloud_img = f"/static/wordclouds/{show_key}/{show_key}_SERIES.png"

    return series_title_summary, season_count, episode_count, scene_count, line_count, word_count, air_date_range, wordcloud_img


############ all series episodes scatter
@dapp_new.callback(
    Output('all-series-episodes-scatter', 'figure'),
    # Output('all-series-episodes-dt', 'children'),
    Input('show-key', 'value'))
    # Input('show-all-series-episodes-dt', 'value'))    
def render_all_series_episodes_scatter(show_key: str):
    print(f'in render_all_series_episodes_scatter, show_key={show_key}')

    season_response = esr.list_seasons(ShowKey(show_key))
    seasons = season_response['seasons']
        
    simple_episodes_response = esr.fetch_simple_episodes(ShowKey(show_key))
    all_episodes = simple_episodes_response['episodes']
    # all_episodes_dict = {episode['episode_key']:episode for episode in all_episodes}
    # all_episodes = list(all_episodes_dict.values())

    # load all episodes into dataframe
    df = pd.DataFrame(all_episodes)
    df['air_date'] = df['air_date'].apply(lambda x: x[:10])

    cols_to_keep = ['episode_key', 'title', 'season', 'sequence_in_season', 'air_date', 'focal_speakers', 'focal_locations', 
                    'topics_universal', 'topics_focused', 'topics_universal_tfidf', 'topics_focused_tfidf']

    df = df[cols_to_keep]

    all_series_episodes_scatter = pscat.build_all_series_episodes_scatter(df, seasons)

    # if 'yes' in show_dt:
    #     # NOTE last-minute first draft effort to sync datatable colors with matplotlib/plotly figure color gradient
    #     display_cols = ['title', 'season', 'episode', 'score', 'rank', 'flattened_topics']
    #     df = df.loc[df['score'] > 0]
    #     df = df.loc[df['rank'] > 0]
    #     df.sort_values('rank', inplace=True, ascending=True)
    #     similar_episode_scores = list(df['score'].values)
    #     viridis_discrete_rgbs = fh.matplotlib_gradient_to_rgb_strings('viridis')
    #     sim_ep_rgbs = fh.map_range_values_to_gradient(similar_episode_scores, viridis_discrete_rgbs)
    #     # sim_ep_rgb_textcolors = {rgb:"Black" for rgb in sim_ep_rgbs}
    #     episode_similarity_dt = cmp.pandas_df_to_dash_dt(df, display_cols, 'rank', sim_ep_rgbs, {}, numeric_precision_overrides={'season': 0, 'episode': 0, 'rank': 0})
    # else: 
    #     episode_similarity_dt = {}

    return all_series_episodes_scatter


############ series speakers gantt callback
@dapp_new.callback(
    Output('series-speakers-gantt', 'figure'),
    Input('show-key', 'value'))    
def render_series_speakers_gantt(show_key: str):
    print(f'in render_series_speakers_gantt, show_key={show_key}')

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
            raise Exception(f'Failure to render_series_gantts: unable to fetch or generate dataframe at file_path={file_path}')
    series_speakers_gantt = pgantt.build_series_gantt(show_key, speaker_gantt_sequence_df, 'speakers')

    return series_speakers_gantt


############ series locations gantt callback
@dapp_new.callback(
    Output('series-locations-gantt', 'figure'),
    Input('show-key', 'value'))    
def render_series_locations_gantt(show_key: str):
    print(f'in render_series_locations_gantt, show_key={show_key}')

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
            raise Exception(f'Failure to render_series_gantts: unable to fetch or generate dataframe at file_path={file_path}')
    series_locations_gantt = pgantt.build_series_gantt(show_key, location_gantt_sequence_df, 'locations')

    return series_locations_gantt


############ series topics gantt callback
@dapp_new.callback(
    Output('series-topics-gantt', 'figure'),
    Input('show-key', 'value'))    
def render_series_topics_gantt(show_key: str):
    print(f'in render_series_topics_gantt, show_key={show_key}')

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
            raise Exception(f'Failure to render_series_gantts: unable to fetch or generate dataframe at file_path={file_path}')
    series_topics_gantt = pgantt.build_series_gantt(show_key, topic_gantt_sequence_df, 'topics')

    return series_topics_gantt


############ series search gantt callback
@dapp_new.callback(
    Output('series-dialog-qt-display', 'children'),
    Output('series-search-results-gantt-new', 'figure'),
    Output('series-search-results-dt', 'children'),
    Input('show-key', 'value'),
    Input('series-dialog-qt', 'value'),
    # NOTE: I believe 'qt-submit' is a placebo: it's a call to action, but simply exiting the qt field invokes the callback
    Input('qt-submit', 'value'))    
def render_series_search_gantt(show_key: str, series_dialog_qt: str, qt_submit: bool = False):
    print(f'in render_series_search_gantt, show_key={show_key} series_dialog_qt={series_dialog_qt} qt_submit={qt_submit}')

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

    if series_dialog_qt:
        series_gantt_response = esr.generate_series_speaker_gantt_sequence(ShowKey(show_key))
        search_response = esr.search_scene_events(ShowKey(show_key), dialog=series_dialog_qt)
        # if 'matches' not in search_response or len(search_response['matches']) == 0:
        #     print(f"no matches for show_key={show_key} qt=`{qt}` qt_submit=`{qt_submit}`")
        #     return None, show_key, qt
        # print(f"len(search_response['matches'])={len(search_response['matches'])}")
        # print(f"len(series_gantt_response['episode_speakers_sequence'])={len(series_gantt_response['episode_speakers_sequence'])}")
        timeline_df = pd.DataFrame(series_gantt_response['episode_speakers_sequence'])
        series_search_results_gantt = pgantt.build_series_search_results_gantt(show_key, timeline_df, search_response['matches'])

        matching_lines_df = timeline_df.loc[timeline_df['matching_line_count'] > 0]

        # build dash datatable
        matching_lines_df.rename(columns={'Task': 'character', 'sequence_in_season': 'episode'}, inplace=True)
        matching_speakers = matching_lines_df['character'].unique()
        speaker_color_map = fh.generate_speaker_color_discrete_map(show_key, matching_speakers)
        # TODO matching_lines_df['dialog'] = matching_lines_df['dialog'].apply(convert_markup)
        display_cols = ['episode_key', 'episode_title', 'count', 'season', 'episode', 'info', 'matching_line_count', 'matching_lines']
        episode_search_results_dt = cmp.pandas_df_to_dash_dt(matching_lines_df, display_cols, 'episode_key', matching_speakers, speaker_color_map, 
                                                            numeric_precision_overrides={'count': 0, 'season': 0, 'episode': 0, 'matching_line_count': 0})

        # out_text = f"{scene_event_count} lines matching query '{qt}'"

    else:
        series_search_results_gantt = {}
        episode_search_results_dt = {}

    return series_dialog_qt, series_search_results_gantt, episode_search_results_dt


############ speaker frequency bar chart callback
@dapp_new.callback(
    Output('speaker-season-frequency-bar-chart', 'figure'),
    Output('speaker-episode-frequency-bar-chart', 'figure'),
    Input('show-key', 'value'),
    Input('span-granularity', 'value'),
    Input('character-chatter-season', 'value'),
    Input('character-chatter-sequence-in-season', 'value'))    
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

    return speaker_season_frequency_bar_chart, speaker_episode_frequency_bar_chart


@dapp_new.callback(
    # Output('speaker-qt-display', 'children'),
    Output('speaker-series-listing-dt', 'children'),
    # Output('speaker-matches-dt', 'children'),
    Input('show-key', 'value'))
    # Input('speaker-qt', 'value')) 
def render_series_speaker_listing_dt(show_key: str):
    print(f'in render_series_speaker_listing_dt, show_key={show_key}')   
# def render_series_speaker_listing_dt(show_key: str, speaker_qt: str):
#     print(f'in render_series_speaker_listing_dt, show_key={show_key} speaker_qt={speaker_qt}')

    indexed_speakers_response = esr.fetch_indexed_speakers(ShowKey(show_key), extra_fields='topics_mbti')
    indexed_speakers = indexed_speakers_response['speakers']
    indexed_speakers = fh.flatten_speaker_topics(indexed_speakers, 'mbti', limit_per_speaker=3) 
    indexed_speakers = fh.flatten_and_refine_alt_names(indexed_speakers, limit_per_speaker=1) 
    
    speakers_df = pd.DataFrame(indexed_speakers)

	# # TODO well THIS is inefficient...
    # indexed_speaker_keys = [s['speaker'] for s in indexed_speakers]
    # speaker_aggs_response = esr.composite_speaker_aggs(show_key)
    # speaker_aggs = speaker_aggs_response['speaker_agg_composite']
    # non_indexed_speakers = [s for s in speaker_aggs if s['speaker'] not in indexed_speaker_keys]

    speaker_names = [s['speaker'] for s in indexed_speakers]

    speakers_df.rename(columns={'speaker': 'character', 'scene_count': 'scenes', 'line_count': 'lines', 'word_count': 'words', 'season_count': 'seasons', 
                                'episode_count': 'episodes', 'actor_names': 'actor(s)', 'topics_mbti': 'mbti'}, inplace=True)
    display_cols = ['character', 'aka', 'actor(s)', 'seasons', 'episodes', 'scenes', 'lines', 'words', 'mbti']

    # replace actor nan values with empty string, flatten list into string
    speakers_df['actor(s)'].fillna('', inplace=True)
    speakers_df['actor(s)'] = speakers_df['actor(s)'].apply(lambda x: ', '.join(x))

    speaker_colors = fh.generate_speaker_color_discrete_map(show_key, speaker_names)

    speaker_listing_dt = cmp.pandas_df_to_dash_dt(speakers_df, display_cols, 'character', speaker_names, speaker_colors,
                                                  numeric_precision_overrides={'seasons': 0, 'episodes': 0, 'scenes': 0, 'lines': 0, 'words': 0})

    # print('speaker_listing_dt:')
    # utils.hilite_in_logs(speaker_listing_dt)

    # speaker_matches = []
    # if speaker_qt:
    # 	# speaker_qt_display = speaker_qt
    #     speaker_search_response = esr.search_speakers(speaker_qt, show_key=show_key)
    #     speaker_matches = speaker_search_response['speaker_matches']
    #     speaker_matches_dt = None

    # return speaker_qt, speaker_listing_dt, speaker_matches_dt
    return speaker_listing_dt


############ series speaker topic grid callbacks
@dapp_new.callback(
    Output('series-speaker-mbti-scatter', 'figure'),
    Output('series-speaker-dnda-scatter', 'figure'),
    Output('series-speaker-mbti-dt', 'children'),
    Output('series-speaker-dnda-dt', 'children'),
    Input('show-key', 'value'),
    Input('series-mbti-count', 'value'),
    Input('series-dnda-count', 'value'))    
def render_series_speaker_topic_scatter(show_key: str, mbti_count: int, dnda_count: int):
    print(f'in render_series_speaker_topic_scatter, show_key={show_key} mbti_count={mbti_count} dnda_count={dnda_count}')

    series_speaker_names = list(show_metadata[show_key]['regular_cast'].keys()) + list(show_metadata[show_key]['recurring_cast'].keys())
    indexed_speakers_response = esr.fetch_indexed_speakers(ShowKey(show_key), extra_fields='topics_mbti,topics_dnda', speakers=','.join(series_speaker_names))
    indexed_speakers = indexed_speakers_response['speakers']
    # indexed_speakers = fh.flatten_speaker_topics(indexed_speakers, 'mbti', limit_per_speaker=3) 

    speaker_color_map = fh.generate_speaker_color_discrete_map(show_key, series_speaker_names)

    # flatten episode speaker topic data for each episode speaker
    exploded_speakers_mbti = fh.explode_speaker_topics(indexed_speakers, 'mbti', limit_per_speaker=mbti_count)
    exploded_speakers_dnda = fh.explode_speaker_topics(indexed_speakers, 'dnda', limit_per_speaker=dnda_count)
    mbti_df = pd.DataFrame(exploded_speakers_mbti)
    dnda_df = pd.DataFrame(exploded_speakers_dnda)
    series_speaker_mbti_scatter = pscat.build_speaker_topic_scatter(show_key, mbti_df.copy(), 'mbti', speaker_color_map=speaker_color_map)
    series_speaker_dnda_scatter = pscat.build_speaker_topic_scatter(show_key, dnda_df.copy(), 'dnda', speaker_color_map=speaker_color_map)

    # build dash datatable
    display_cols = ['speaker', 'topic_key', 'topic_name', 'score', 'raw_score']
    series_speaker_mbti_dt = cmp.pandas_df_to_dash_dt(mbti_df, display_cols, 'speaker', series_speaker_names, speaker_color_map)
    series_speaker_dnda_dt = cmp.pandas_df_to_dash_dt(dnda_df, display_cols, 'speaker', series_speaker_names, speaker_color_map)

    return series_speaker_mbti_scatter, series_speaker_dnda_scatter, series_speaker_mbti_dt, series_speaker_dnda_dt


############ series topic pie and bar chart callbacks
@dapp_new.callback(
    Output('series-topic-pie', 'figure'),
    Output('series-parent-topic-pie', 'figure'),
    # Output('series-topic-bar', 'figure'),
    # Output('series-parent-topic-bar', 'figure'),
    Input('show-key', 'value'),
    Input('topic-grouping', 'value'),
    Input('score-type', 'value'))    
def render_series_topic_figs(show_key: str, topic_grouping: str, score_type: str):
    print(f'in render_series_topic_figs, show_key={show_key} topic_grouping={topic_grouping} score_type={score_type}')

    episode_response = esr.fetch_simple_episodes(ShowKey(show_key))
    episode_topic_lists = []
    for episode in episode_response['episodes']:
        episode_topics_response = esr.fetch_episode_topics(ShowKey(show_key), episode['episode_key'], topic_grouping)
        episode_topic_lists.append(episode_topics_response['episode_topics'])

    series_topics_df, series_parent_topics_df = fh.generate_topic_aggs_from_episode_topics(episode_topic_lists, max_rank=20, max_parent_repeats=2)

    series_topics_pie = ppie.build_topic_aggs_pie(series_topics_df, topic_grouping, score_type)
    series_parent_topics_pie = ppie.build_topic_aggs_pie(series_parent_topics_df, topic_grouping, score_type, is_parent=True)

    return series_topics_pie, series_parent_topics_pie


############ series topic episode datatable callbacks
@dapp_new.callback(
    Output('series-topic-episodes-dt', 'children'),
    Input('show-key', 'value'),
    Input('topic-grouping', 'value'),
    Input('parent-topic', 'value'),
    Input('score-type', 'value'))    
def render_series_topic_episodes_dt(show_key: str, topic_grouping: str, parent_topic: str, score_type: str):
    print(f'in render_series_topic_episodes_dt, show_key={show_key} topic_grouping={topic_grouping} parent_topic={parent_topic} score_type={score_type}')

    # NOTE assembling entire parent-child topic hierarchy here, but only using one branch of the tree
    child_topics = []
    topic_grouping_response = esr.fetch_topic_grouping(topic_grouping)
    for t in topic_grouping_response['topics']:
        # only process topics that have parents (ignore the parents themselves)
        if not t['parent_key']:
            continue
        if parent_topic == t['topic_key'].split('.')[0]:
            child_topics.append(t['topic_key'])

    if not child_topics:
        utils.hilite_in_logs(f'Failure to render_series_topic_episodes_dt, no child topics for parent_topic={parent_topic} in topic_grouping={topic_grouping}')
        return {}

    columns = ['topic_key', 'parent_topic', 'episode_key', 'episode_title', 'season', 'sequence_in_season', 'air_date', 'score', 'tfidf_score']
    topic_episodes_df = pd.DataFrame(columns=columns)
    # for parent_topic, child_topics in parent_to_leaf_topics.items():
    for topic in child_topics:
        episodes_by_topic = esr.find_episodes_by_topic(ShowKey(show_key), topic_grouping, topic)
        df = pd.DataFrame(episodes_by_topic['episode_topics'])
        df['parent_topic'] = parent_topic
        df = df[columns]
        df = df[(df['score'] > 0.5) | (df['tfidf_score'] > 0.5)]
        topic_episodes_df = pd.concat([topic_episodes_df, df])

    topic_episodes_df.sort_values(score_type, ascending=False, inplace=True)

    series_topic_episodes_dt = cmp.pandas_df_to_dash_dt(topic_episodes_df, columns, 'parent_topic', [parent_topic], TOPIC_COLORS)

    return series_topic_episodes_dt


if __name__ == "__main__":
    dapp_new.run_server(debug=True)
