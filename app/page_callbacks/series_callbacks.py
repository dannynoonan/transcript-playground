from dash import callback, Input, Output
from datetime import datetime as dt
import os
import pandas as pd

import app.es.es_read_router as esr
import app.es.es_query_builder as esqb
import app.es.es_response_transformer as esrt
import app.fig_builder.plotly_bar as pbar
import app.fig_builder.plotly_gantt as pgantt
import app.fig_builder.plotly_pie as ppie
import app.fig_builder.plotly_scatter as pscat
import app.fig_meta.color_meta as cm
import app.fig_meta.gantt_helper as gh
import app.data_service.field_flattener as fflat
import app.data_service.topic_aggregator as tagg
import app.nlp.embeddings_factory as ef
import app.page_builder_service.page_components as pc
import app.page_builder_service.series_page_service as sps
from app.show_metadata import show_metadata, ShowKey
from app import utils


############ series summary callbacks
@callback(
    Output("accordion-contents", "children"),
    Input('show-key', 'data'),
    Input("accordion", "active_item")
)    
def render_series_summary(show_key: str, expanded_season: str):
    utils.hilite_in_logs(f'callback invoked: render_series_summary, show_key={show_key} expanded_season={expanded_season}')

    # TODO circle back to whether this is needed and how to label it
    # accordion_contents = {}

    return {}


############ all series episodes scatter
@callback(
    Output('series-episodes-scatter-grid', 'figure'),
    Input('show-key', 'data'),
    Input('scatter-grid-hilite', 'value'),
    Input('speaker-color-map', 'data'),
    Input('all-simple-episodes', 'data'),
    Input('all-seasons', 'data'),
    background=True
)    
def render_all_series_episodes_scatter(show_key: str, hilite: str, speaker_color_map: dict, all_simple_episodes: list, all_seasons: list):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_all_series_episodes_scatter ts={callback_start_ts} show_key={show_key} hilite={hilite}')

    if hilite in ['topics_universal', 'topics_universal_tfidf']:
        hilite_color_map = cm.TOPIC_COLORS
    elif hilite == 'focal_speakers':
        hilite_color_map = speaker_color_map
    elif hilite == 'focal_locations':
        scenes_by_location_response = esr.agg_scenes_by_location(ShowKey(show_key))
        scenes_by_location = scenes_by_location_response['scenes_by_location']
        locations = utils.truncate_dict(scenes_by_location, 500, start_index=1)
        hilite_color_map = {loc:cm.colors[i % 10] for i, loc in enumerate(locations)}
    else:
        hilite_color_map = None

    # load all episodes into dataframe
    df = pd.DataFrame(all_simple_episodes)
    df['air_date'] = df['air_date'].apply(lambda x: x[:10])

    cols_to_keep = ['episode_key', 'title', 'season', 'sequence_in_season', 'air_date', 'focal_speakers', 'focal_locations', 
                    'topics_universal', 'topics_universal_tfidf']

    df = df[cols_to_keep]

    all_series_episodes_scatter = pscat.build_all_series_episodes_scatter(df, all_seasons, hilite=hilite, hilite_color_map=hilite_color_map)

    callback_end_ts = dt.now()
    callback_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_all_series_episodes_scatter returned at ts={callback_end_ts} duration={callback_duration}')

    return all_series_episodes_scatter


############ series speakers gantt callback
@callback(
    Output('series-speakers-gantt', 'figure'),
    Input('show-key', 'data'),
    Input('simple-episodes-by-season', 'data'),
    background=True
)    
def render_series_speakers_gantt(show_key: str, simple_episodes_by_season: dict):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_speakers_gantt ts={callback_start_ts} show_key={show_key}')

    season_interval_data = gh.simple_season_episode_i_map(simple_episodes_by_season)

    file_path = f'./app/data/{show_key}/speaker_gantt_sequence_{show_key}.csv'
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
            raise Exception(f'Failure to render_series_speakers_gantt: unable to fetch or generate dataframe at file_path={file_path}')

    series_speakers_gantt = pgantt.build_series_gantt(show_key, speaker_gantt_sequence_df, 'speakers', interval_data=season_interval_data)

    callback_end_ts = dt.now()
    callback_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_speakers_gantt returned at ts={callback_end_ts} duration={callback_duration}')

    return series_speakers_gantt


############ series locations gantt callback
@callback(
    Output('series-locations-gantt', 'figure'),
    Input('show-key', 'data'),
    Input('simple-episodes-by-season', 'data'),
    background=True
)    
def render_series_locations_gantt(show_key: str, simple_episodes_by_season: dict):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_locations_gantt ts={callback_start_ts} show_key={show_key}')

    season_interval_data = gh.simple_season_episode_i_map(simple_episodes_by_season)

    file_path = f'./app/data/{show_key}/location_gantt_sequence_{show_key}.csv'
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
            raise Exception(f'Failure to render_series_locations_gantt: unable to fetch or generate dataframe at file_path={file_path}')
        
    series_locations_gantt = pgantt.build_series_gantt(show_key, location_gantt_sequence_df, 'locations', interval_data=season_interval_data)

    callback_end_ts = dt.now()
    callback_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_locations_gantt returned at ts={callback_end_ts} duration={callback_duration}')

    return series_locations_gantt


############ series topics gantt callback
@callback(
    Output('series-topics-gantt', 'figure'),
    Input('show-key', 'data'),
    Input('series-topics-gantt-score-type', 'value'),
    Input('simple-episodes-by-season', 'data'),
    background=True
)    
def render_series_topics_gantt(show_key: str, score_type: str, simple_episodes_by_season: dict):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_topics_gantt ts={callback_start_ts} show_key={show_key}')

    season_interval_data = gh.simple_season_episode_i_map(simple_episodes_by_season)

    topic_grouping = 'universalGenres'
    topic_threshold = 20
    file_path = f'./app/data/{show_key}/topic_gantt_sequence_{show_key}_{topic_grouping}_{score_type}.csv'
    if os.path.isfile(file_path):
        topic_gantt_sequence_df = pd.read_csv(file_path)
        print(f'loading dataframe at file_path={file_path}')
    else:
        print(f'no file found at file_path={file_path}, running `/esr/generate_series_topic_gantt_sequence/{show_key}?overwrite_file=True` with params to generate')
        esr.generate_series_topic_gantt_sequence(ShowKey(show_key), overwrite_file=True, topic_grouping=topic_grouping, 
                                                 topic_threshold=topic_threshold, score_type=score_type)
        if os.path.isfile(file_path):
            topic_gantt_sequence_df = pd.read_csv(file_path)
            print(f'loading dataframe at file_path={file_path}')
        else:
            raise Exception(f'Failure to render_series_gantts: unable to fetch or generate dataframe at file_path={file_path}')
        
    series_topics_gantt = pgantt.build_series_gantt(show_key, topic_gantt_sequence_df, 'topics', interval_data=season_interval_data)

    callback_end_ts = dt.now()
    callback_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_topics_gantt returned at ts={callback_end_ts} duration={callback_duration}')

    return series_topics_gantt


############ series search gantt callback
@callback(
    Output('series-search-response-text', 'children'),
    Output('series-search-results-gantt-new', 'figure'),
    Output('series-search-results-dt', 'children'),
    Input('show-key', 'data'),
    Input('series-search-qt', 'value'),
    Input('simple-episodes-by-season', 'data'),
)    
def render_series_search_gantt(show_key: str, qt: str, simple_episodes_by_season: dict):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_search_gantt ts={callback_start_ts} show_key={show_key} series_dialog_qt={qt}')

    season_interval_data = gh.simple_season_episode_i_map(simple_episodes_by_season)

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

    if not qt:
        return '', {}, {}
    
    # execute search query and filter response into series gantt charts
    series_gantt_response = esr.generate_series_speaker_gantt_sequence(ShowKey(show_key))
    search_response = esr.search_scene_events(ShowKey(show_key), dialog=qt)
    episode_count = search_response['episode_count']
    scene_event_count = search_response['scene_event_count']
    response_text = f"{scene_event_count} line(s) in {episode_count} episode(s) matching query '{qt}'"

    timeline_df = pd.DataFrame(series_gantt_response['episode_speakers_sequence'])
    series_search_results_gantt = pgantt.build_series_search_results_gantt(show_key, timeline_df, search_response['matches'], interval_data=season_interval_data)

    # build dash datatable
    series_search_results_dt = sps.generate_series_search_results_dt(show_key, timeline_df)

    callback_end_ts = dt.now()
    callback_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_search_gantt returned at ts={callback_end_ts} duration={callback_duration}')

    return response_text, series_search_results_gantt, series_search_results_dt


############ speaker frequency bar chart callback
@callback(
    Output('speaker-season-frequency-bar-chart', 'figure'),
    Output('speaker-episode-frequency-bar-chart', 'figure'),
    Input('show-key', 'data'),
    Input('speaker-chatter-tally-by', 'value'),
    Input('speaker-chatter-season', 'value'),
    Input('speaker-chatter-sequence-in-season', 'value')
)    
def render_speaker_frequency_bar_chart(show_key: str, tally_by: str, season: str, sequence_in_season: str = None):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_speaker_frequency_bar_chart ts={callback_start_ts} show_key={show_key} tally_by={tally_by} season={season} sequence_in_season={sequence_in_season}')

    if season in ['0', 0, 'All']:
        season = None
    else:
        season = int(season)

    # fetch or generate aggregate speaker data and build speaker frequency bar chart
    file_path = f'./app/data/{show_key}/speaker_episode_aggs_{show_key}.csv'
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
            raise Exception(f'Failure to render_speaker_frequency_bar_chart: unable to fetch or generate dataframe at file_path={file_path}')
    
    speaker_season_frequency_bar_chart = pbar.build_speaker_frequency_bar(show_key, df, tally_by, aggregate_ratio=False, season=season)
    speaker_episode_frequency_bar_chart = pbar.build_speaker_frequency_bar(show_key, df, tally_by, aggregate_ratio=False, season=season, sequence_in_season=sequence_in_season)

    callback_end_ts = dt.now()
    callback_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_speaker_frequency_bar_chart returned at ts={callback_end_ts} duration={callback_duration}')

    return speaker_season_frequency_bar_chart, speaker_episode_frequency_bar_chart


############ series topic pie and bar chart callback
@callback(
    Output('series-topic-pie', 'figure'),
    Output('series-parent-topic-pie', 'figure'),
    Input('show-key', 'data'),
    Input('series-topic-pie-topic-grouping', 'value'),
    Input('series-topic-pie-score-type', 'value'),
    Input('all-simple-episodes', 'data'),
    background=True
)    
def render_series_topic_pies(show_key: str, topic_grouping: str, score_type: str, all_simple_episodes: str):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_topic_pies ts={callback_start_ts} show_key={show_key} topic_grouping={topic_grouping} score_type={score_type}')

    ##### TODO begin optimization block 
    episode_topic_lists = []
    for episode in all_simple_episodes:
        episode_topics_response = esr.fetch_episode_topics(ShowKey(show_key), episode['episode_key'], topic_grouping)
        episode_topic_lists.append(episode_topics_response['episode_topics'])

    series_topics_df, series_parent_topics_df = tagg.generate_topic_aggs_from_episode_topics(episode_topic_lists, max_rank=20, max_parent_repeats=2)
    ##### TODO end optimization block 

    series_topics_pie = ppie.build_topic_aggs_pie(series_topics_df, topic_grouping, score_type)
    series_parent_topics_pie = ppie.build_topic_aggs_pie(series_parent_topics_df, topic_grouping, score_type, is_parent=True)

    callback_end_ts = dt.now()
    callback_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_topic_pies returned at ts={callback_end_ts} duration={callback_duration}')

    return series_topics_pie, series_parent_topics_pie


# NOTE I'm not sure why this isn't just part of `render_series_topic_pies` callback (or maybe all optional dts should have separate callbacks?)
############ series topic episode datatable callback
@callback(
    Output('series-topic-episodes-dt', 'children'),
    Input('show-key', 'data'),
    Input('display-episodes-dt-for-topic', 'value'),
    Input('series-topic-pie-topic-grouping', 'value'),
    Input('series-topic-pie-score-type', 'value')
)    
def render_series_topic_episodes_dt(show_key: str, display_dt_for_topic: str, topic_grouping: str, score_type: str):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_topic_episodes_dt ts={callback_start_ts} show_key={show_key} display_dt_for_topic={display_dt_for_topic} topic_grouping={topic_grouping} score_type={score_type}')

    if not display_dt_for_topic:
        return {}
    
    # configurable score threshold
    min_score = 0.5

    # NOTE assembling entire parent-child topic hierarchy here, but only using one branch of the tree
    child_topics = []
    topic_grouping_response = esr.fetch_topic_grouping(topic_grouping)
    for t in topic_grouping_response['topics']:
        # only process topics that have parents (ignore the parents themselves)
        if not t['parent_key']:
            continue
        if display_dt_for_topic == t['topic_key'].split('.')[0]:
            child_topics.append(t['topic_key'])

    if not child_topics:
        child_topics = [display_dt_for_topic]

    columns = ['topic_key', 'parent_topic', 'episode_key', 'episode_title', 'season', 'sequence_in_season', 'air_date', 'score', 'tfidf_score']
    topic_episodes_df = pd.DataFrame(columns=columns)
    # for parent_topic, child_topics in parent_to_leaf_topics.items():
    for topic in child_topics:
        episodes_by_topic = esr.find_episodes_by_topic(ShowKey(show_key), topic_grouping, topic)
        df = pd.DataFrame(episodes_by_topic['episode_topics'])
        df['parent_topic'] = display_dt_for_topic
        df = df[columns]
        df = df[(df['score'] > min_score) | (df['tfidf_score'] > min_score)]
        topic_episodes_df = pd.concat([topic_episodes_df, df])

    series_topic_episodes_dt = sps.generate_series_topic_episodes_dt(show_key, topic_episodes_df, display_dt_for_topic, score_type)

    callback_end_ts = dt.now()
    callback_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_topic_episodes_dt returned at ts={callback_end_ts} duration={callback_duration}')

    return series_topic_episodes_dt


############ series cluster scatter callback
@callback(
    Output('series-episodes-cluster-scatter', 'figure'),
    Output('series-episodes-cluster-dt', 'children'),
    Input('show-key', 'data'),
    Input('num-clusters', 'value'),
    Input('display-series-episodes-cluster-dt', 'value'),
    Input('all-simple-episodes', 'data')
    # background=True
)
def render_series_cluster_scatter(show_key: str, num_clusters: int, display_dt: list, all_simple_episodes: list):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_cluster_scatter ts={callback_start_ts} show_key={show_key} num_clusters={num_clusters} display_dt={display_dt}')

    num_clusters = int(num_clusters)
    vector_field = 'openai_ada002_embeddings'

    # fetch embeddings for all show episodes 
    s = esqb.fetch_series_embeddings(show_key, vector_field)
    doc_embeddings = esrt.return_all_embeddings(s, vector_field)

    # generate and color-stamp clusters for all show episodes 
    doc_embeddings_clusters_df = ef.cluster_docs(doc_embeddings, num_clusters)
    doc_embeddings_clusters_df['cluster_color'] = doc_embeddings_clusters_df['cluster'].apply(lambda x: cm.colors[x % 10])

    # fetch basic title/season data for all show episodes 
    episodes_df = pd.DataFrame(all_simple_episodes)

    # merge basic episode data into cluster data
    episodes_df['doc_id'] = episodes_df['episode_key'].apply(lambda x: f'{show_key}_{x}')
    episode_embeddings_clusters_df = pd.merge(doc_embeddings_clusters_df, episodes_df, on='doc_id', how='outer')

    # generate scatterplot
    episode_clusters_scatter = pscat.build_cluster_scatter(episode_embeddings_clusters_df, show_key, num_clusters)

    if 'yes' in display_dt:
        episode_clusters_dt = sps.generate_series_clusters_dt(show_key, episode_embeddings_clusters_df)
    else:
        episode_clusters_dt = {}

    callback_end_ts = dt.now()
    callback_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_cluster_scatter returned at ts={callback_end_ts} duration={callback_duration}')

    return episode_clusters_scatter, episode_clusters_dt


############ series speaker listing callback
@callback(
    Output('series-speaker-listing-dt', 'children'),
    Input('show-key', 'data'),
    Input('indexed-speakers', 'data')
)
def render_series_speaker_listing_dt(show_key: str, indexed_speakers: list):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_speaker_listing_dt ts={callback_start_ts} show_key={show_key}')

    indexed_speakers = fflat.flatten_speaker_topics(indexed_speakers, 'mbti', limit_per_speaker=3) 
    indexed_speakers = fflat.flatten_and_refine_alt_names(indexed_speakers, limit_per_speaker=1) 
    
    speakers_df = pd.DataFrame(indexed_speakers)
    speaker_listing_dt = sps.generate_series_speaker_listing_dt(show_key, speakers_df, indexed_speakers)

    callback_end_ts = dt.now()
    callback_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_speaker_listing_dt returned at ts={callback_end_ts} duration={callback_duration}')

    # return speaker_qt, speaker_listing_dt, speaker_matches_dt
    return speaker_listing_dt


############ series speaker topic grid callback
@callback(
    Output('series-speaker-mbti-scatter', 'figure'),
    Output('series-speaker-dnda-scatter', 'figure'),
    Output('series-speaker-mbti-dt', 'children'),
    Output('series-speaker-dnda-dt', 'children'),
    Input('show-key', 'data'),
    Input('series-mbti-count', 'value'),
    Input('series-dnda-count', 'value'),
    Input('indexed-speakers', 'data'),
    Input('speaker-color-map', 'data')
)    
def render_series_speaker_topic_scatter(show_key: str, mbti_count: int, dnda_count: int, indexed_speakers: list, speaker_color_map: dict):
    callback_start_ts = dt.now()
    utils.hilite_in_logs(f'callback invoked: render_series_speaker_topic_scatter ts={callback_start_ts} show_key={show_key} mbti_count={mbti_count} dnda_count={dnda_count}')

    speaker_names = [s['speaker'] for s in indexed_speakers]

    # flatten episode speaker topic data for each episode speaker
    exploded_speakers_mbti = fflat.explode_speaker_topics(indexed_speakers, 'mbti', limit_per_speaker=mbti_count)
    exploded_speakers_dnda = fflat.explode_speaker_topics(indexed_speakers, 'dnda', limit_per_speaker=dnda_count)
    mbti_df = pd.DataFrame(exploded_speakers_mbti)
    dnda_df = pd.DataFrame(exploded_speakers_dnda)
    series_speaker_mbti_scatter = pscat.build_speaker_topic_scatter(show_key, mbti_df.copy(), 'mbti', speaker_color_map=speaker_color_map)
    series_speaker_dnda_scatter = pscat.build_speaker_topic_scatter(show_key, dnda_df.copy(), 'dnda', speaker_color_map=speaker_color_map)

    # build dash datatable
    display_cols = ['speaker', 'topic_key', 'topic_name', 'score']
    numeric_precision_overrides = {'score': 2}
    series_speaker_mbti_dt = pc.pandas_df_to_dash_dt(mbti_df, display_cols, 'speaker', speaker_names, speaker_color_map, 
                                                     numeric_precision_overrides=numeric_precision_overrides)
    series_speaker_dnda_dt = pc.pandas_df_to_dash_dt(dnda_df, display_cols, 'speaker', speaker_names, speaker_color_map,
                                                     numeric_precision_overrides=numeric_precision_overrides)

    callback_end_ts = dt.now()
    callback_duration = callback_end_ts - callback_start_ts
    utils.hilite_in_logs(f'render_series_speaker_topic_scatter returned at ts={callback_end_ts} duration={callback_duration}')

    return series_speaker_mbti_scatter, series_speaker_dnda_scatter, series_speaker_mbti_dt, series_speaker_dnda_dt
