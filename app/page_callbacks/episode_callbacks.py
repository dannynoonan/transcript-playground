from dash import callback, Input, Output
from operator import itemgetter
import os
import pandas as pd

import app.es.es_read_router as esr
import app.fig_builder.plotly_bar as pbar
import app.fig_builder.plotly_gantt as pgantt
import app.fig_builder.plotly_line as pline
import app.fig_builder.plotly_networkgraph as pgraph
import app.fig_builder.plotly_scatter as pscat
import app.fig_builder.plotly_treemap as ptree
import app.fig_meta.color_meta as cm
import app.data_service.field_flattener as fflat
from app.nlp.nlp_metadata import OPENAI_EMOTIONS
import app.page_builder_service.page_components as pc
from app.show_metadata import ShowKey
from app import utils


############ episode summary callbacks
@callback(
    Output('episode-title-summary', 'children'),
    Output('episode-scene-count', 'children'),
    Output('episode-line-count', 'children'),
    Output('episode-word-count', 'children'),
    Output('episode-focal-speakers', 'children'),
    Output('episode-topics', 'children'),
    Output('episode-speakers', 'options'),
    Output('episode-wordcloud-img', 'src'),
    Input('show-key', 'value'),
    Input('episode-key', 'value')
)    
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
@callback(
    Output('episode-dialog-timeline-new', 'figure'),
    Output('episode-location-timeline-new', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('show-layers', 'value')
)    
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
@callback(
    Output('episode-search-response-text', 'children'),
    Output('episode-search-results-gantt', 'figure'),
    Output('episode-search-results-dt', 'children'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('episode-search-qt', 'value')
)    
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
    speaker_color_map = cm.generate_speaker_color_discrete_map(show_key, matching_speakers)
    # TODO matching_lines_df['dialog'] = matching_lines_df['dialog'].apply(convert_markup)
    display_cols = ['character', 'scene', 'line', 'location', 'dialog']
    episode_search_results_dt = pc.pandas_df_to_dash_dt(matching_lines_df, display_cols, 'character', matching_speakers, speaker_color_map,
                                                        numeric_precision_overrides={'scene': 0, 'line': 0})

    response_text = f"{scene_event_count} line(s) matching query '{qt}'"

    return response_text, episode_search_results_gantt, episode_search_results_dt


############ sentiment line chart callbacks
@callback(
    Output('sentiment-line-chart-new', 'figure'),
    # Output('episode-speaker-options', 'options'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('freeze-on', 'value'),
    Input('emotion', 'value'),
    Input('episode-speakers', 'value')
)    
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
@callback(
    Output('speaker-3d-network-graph-new', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('scale-by', 'value')
)    
def render_speaker_3d_network_graph_new(show_key: str, episode_key: str, scale_by: str):
    print(f'in render_speaker_3d_network_graph_new, show_key={show_key} episode_key={episode_key} scale_by={scale_by}')

    # generate speaker relations data and build 3d network graph
    speaker_relations_data = esr.speaker_relations_graph(ShowKey(show_key), episode_key)

    # NOTE where and how to layer in color mapping is a WIP
    speakers = [n['speaker'] for n in speaker_relations_data['nodes']]
    speaker_colors = cm.generate_speaker_color_discrete_map(show_key, speakers)
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
@callback(
    Output('speaker-episode-frequency-bar-chart-new', 'figure'),
    Output('speaker-episode-summary-dt', 'children'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('scale-by', 'value')
)    
def render_speaker_frequency_bar_chart_new(show_key: str, episode_key: str, scale_by: str):
    print(f'in render_speaker_frequency_bar_chart_new, show_key={show_key} episode_key={episode_key} scale_by={scale_by}')

    speakers_for_episode_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key, extra_fields='topics_mbti')
    speakers_for_episode = speakers_for_episode_response['speaker_episodes']
    speakers_for_episode = fflat.flatten_speaker_topics(speakers_for_episode, 'mbti', 3)

    df = pd.DataFrame(speakers_for_episode, columns=['speaker', 'agg_score', 'scene_count', 'line_count', 'word_count', 'topics_mbti'])

    episode_speaker_names = [s['speaker'] for s in speakers_for_episode]
    speaker_color_map = cm.generate_speaker_color_discrete_map(show_key, episode_speaker_names)

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

    speaker_episode_summary_dt = pc.pandas_df_to_dash_dt(df, df.columns, 'character', episode_speaker_names, speaker_color_map, 
                                                         numeric_precision_overrides={'agg_score': 2})
        
    return speaker_episode_frequency_bar_chart, speaker_episode_summary_dt


############ episode similarity scatter callbacks
@callback(
    Output('episode-similarity-scatter', 'figure'),
    Output('episode-similarity-dt', 'children'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('mlt-type', 'value'),
    Input('show-similar-episodes-dt', 'value')
)    
def render_episode_similarity_scatter(show_key: str, episode_key: str, mlt_type: str, show_dt: list):
    print(f'in render_episode_similarity_scatter, show_key={show_key} episode_key={episode_key} mlt_type={mlt_type} show_dt={show_dt}')

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
        viridis_discrete_rgbs = cm.matplotlib_gradient_to_rgb_strings('viridis')
        sim_ep_rgbs = cm.map_range_values_to_gradient(similar_episode_scores, viridis_discrete_rgbs)
        # sim_ep_rgb_textcolors = {rgb:"Black" for rgb in sim_ep_rgbs}
        episode_similarity_dt = pc.pandas_df_to_dash_dt(df, display_cols, 'rank', sim_ep_rgbs, {}, numeric_precision_overrides={'score': 2})
    else: 
        episode_similarity_dt = {}

    return episode_similarity_scatter, episode_similarity_dt


############ episode speaker topic grid callbacks
@callback(
    Output('episode-speaker-mbti-scatter', 'figure'),
    Output('episode-speaker-dnda-scatter', 'figure'),
    Output('episode-speaker-mbti-dt', 'children'),
    Output('episode-speaker-dnda-dt', 'children'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('episode-mbti-count', 'value'),
    Input('episode-dnda-count', 'value')
)    
def render_episode_speaker_topic_scatter(show_key: str, episode_key: str, mbti_count: int, dnda_count: int):
    print(f'in render_episode_speaker_topic_scatter, show_key={show_key} episode_key={episode_key} mbti_count={mbti_count} dnda_count={dnda_count}')

    # fetch episode speakers
    speakers_for_episode_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key, extra_fields='topics_mbti,topics_dnda')
    episode_speakers = speakers_for_episode_response['speaker_episodes']
    
    # fetched series-level indexed version of episode speakers
    episode_speaker_names = [s['speaker'] for s in episode_speakers]
    # indexed_speakers_response = esr.fetch_indexed_speakers(ShowKey(show_key), extra_fields='topics_mbti,topics_dnda', speakers=','.join(episode_speaker_names))

    speaker_color_map = cm.generate_speaker_color_discrete_map(show_key, episode_speaker_names)

    # NOTE ended up not using this data downstream
    # mbti_distribution_response = esr.agg_numeric_distrib_into_percentiles(ShowKey(show_key), 'speaker_episode_topics', 'raw_score', constraints='topic_grouping:meyersBriggsKiersey')
    # dnda_distribution_response = esr.agg_numeric_distrib_into_percentiles(ShowKey(show_key), 'speaker_episode_topics', 'raw_score', constraints='topic_grouping:dndAlignments')

    # mbti_percent_distrib = mbti_distribution_response["percentile_distribution"]
    # mbti_percent_distrib_list = list(mbti_percent_distrib.values())
    # dnda_percent_distrib = dnda_distribution_response["percentile_distribution"]
    # dnda_percent_distrib_list = list(dnda_percent_distrib.values())

    # flatten episode speaker topic data for each episode speaker
    exploded_speakers_mbti = fflat.explode_speaker_topics(episode_speakers, 'mbti', limit_per_speaker=mbti_count)
    exploded_speakers_dnda = fflat.explode_speaker_topics(episode_speakers, 'dnda', limit_per_speaker=dnda_count)
    mbti_df = pd.DataFrame(exploded_speakers_mbti)
    dnda_df = pd.DataFrame(exploded_speakers_dnda)
    episode_speaker_mbti_scatter = pscat.build_speaker_topic_scatter(show_key, mbti_df.copy(), 'mbti', speaker_color_map=speaker_color_map)
    episode_speaker_dnda_scatter = pscat.build_speaker_topic_scatter(show_key, dnda_df.copy(), 'dnda', speaker_color_map=speaker_color_map)

    # build dash datatable
    display_cols = ['speaker', 'topic_key', 'topic_name', 'score']
    numeric_precision_overrides = {'score': 2}
    episode_speaker_mbti_dt = pc.pandas_df_to_dash_dt(mbti_df, display_cols, 'speaker', episode_speaker_names, speaker_color_map, 
                                                      numeric_precision_overrides=numeric_precision_overrides)
    episode_speaker_dnda_dt = pc.pandas_df_to_dash_dt(dnda_df, display_cols, 'speaker', episode_speaker_names, speaker_color_map,
                                                      numeric_precision_overrides=numeric_precision_overrides)
    return episode_speaker_mbti_scatter, episode_speaker_dnda_scatter, episode_speaker_mbti_dt, episode_speaker_dnda_dt


############ episode topic treemap callbacks
@callback(
    Output('episode-universal-genres-treemap', 'figure'),
    Output('episode-universal-genres-dt', 'children'),
    Output('episode-universal-genres-gpt35-v2-treemap', 'figure'),
    Output('episode-universal-genres-gpt35-v2-dt', 'children'),
    # Output('episode-focused-gpt35-treemap', 'figure'),
    Input('show-key', 'value'),
    Input('episode-key', 'value'),
    Input('universal-genres-score-type', 'value'),
    Input('universal-genres-gpt35-v2-score-type', 'value')
)    
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
        df = fflat.flatten_and_format_topics_df(df, topic_score_types[i])
        # build treemap fig
        fig = ptree.build_episode_topic_treemap(df.copy(), tg, topic_score_types[i], max_per_parent=3)
        figs[tg] = fig
        # build dash datatable
        parent_topics = df['parent_topic'].unique()
        df.rename(columns={'scaled_score': 'score'}, inplace=True)
        display_cols = ['parent_topic', 'topic_name', 'score', 'tfidf_score']
        dash_dt = pc.pandas_df_to_dash_dt(df, display_cols, 'parent_topic', parent_topics, cm.TOPIC_COLORS,
                                          numeric_precision_overrides={'score': 2, 'tfidf_score': 2})
        dts[tg] = dash_dt

    return figs['universalGenres'], dts['universalGenres'], figs['universalGenresGpt35_v2'], dts['universalGenresGpt35_v2']


# # NOTE not being used
# ############ episode speaker chatter scatter callbacks
# @callback(
#     Output('speaker-chatter-scatter', 'figure'),
#     Input('show-key', 'value'),
#     Input('episode-key', 'value'),
#     Input('x-axis', 'value'),
#     Input('y-axis', 'value')
# )    
# def render_speaker_chatter_scatter(show_key: str, episode_key: str, x_axis: str, y_axis: str):
#     print(f'in render_speaker_chatter_scatter, show_key={show_key} episode_key={episode_key} x_axis={x_axis} y_axis={y_axis}')

#     speakers_for_episode_response = esr.fetch_speakers_for_episode(ShowKey(show_key), episode_key)
#     speakers_for_episode = speakers_for_episode_response['speaker_episodes']
#     df = pd.DataFrame(speakers_for_episode, columns=['speaker', 'agg_score', 'scene_count', 'line_count', 'word_count'])
    
#     speaker_chatter_scatter = pscat.build_speaker_chatter_scatter(df, x_axis, y_axis)

#     return speaker_chatter_scatter
