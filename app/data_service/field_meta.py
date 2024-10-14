from app.es.es_metadata import SENTIMENT_FIELDS, TOPICS_FIELDS, FOCAL_FIELDS


# TODO happened upon these in Aug 2024, not 100% of their usage but I suspect es query response filters/flags would handle this better
# episode_keep_cols = ['title', 'season', 'sequence_in_season', 'air_date', 'focal_speakers', 'focal_locations', 'scene_count', 'episode_key']
episode_keep_cols = ['title', 'season', 'sequence_in_season', 'air_date', 'scene_count', 'episode_key']
# episode_drop_cols = ['doc_id', 'show_key', 'indexed_ts', 'topics_universal', 'topics_focused', 'topics_universal_tfidf', 'topics_focused_tfidf']
episode_drop_cols = ['doc_id', 'show_key', 'indexed_ts'] + SENTIMENT_FIELDS + TOPICS_FIELDS + FOCAL_FIELDS
cluster_cols = ['cluster', 'cluster_color']
