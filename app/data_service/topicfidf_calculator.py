import math

from app.auth import ADMIN_USER
import app.es.es_query_builder as esqb
import app.routers.es_read_router as esr
from app.show_metadata import ShowKey


def calculate_topic_freq_idf(show_key: ShowKey, topic_grouping: str, model_vendor: str, model_version: str) -> tuple[dict, dict]:

    all_topics_response = esr.fetch_topic_grouping(topic_grouping, ADMIN_USER)
    # topic_agg_scores is a stand-in for "document frequency"
    topic_agg_scores = {t['topic_key']:0 for t in all_topics_response['topics']}

    simple_episodes_response = esr.fetch_simple_episodes(show_key, ADMIN_USER)
    e_keys = [e['episode_key'] for e in simple_episodes_response['episodes']]
    ekey_tkey_scores = {ek:{} for ek in e_keys}

    # TODO replace with agg query? since we're not actually fetching the `episode_topic` entities by id to update them
    for e_key in e_keys:
        episode_topics_response = esr.fetch_episode_topics(show_key, e_key, topic_grouping, model_vendor, model_version, ADMIN_USER)
        episode_topics = episode_topics_response['episode_topics']
        # topic_agg_scores is a stand-in for "document frequency"
        for topic in episode_topics:
            t_key = topic['topic_key']
            t_score = topic['score']
            topic_agg_scores[t_key] += t_score
            ekey_tkey_scores[e_key][t_key] = t_score

    # use topic_agg_scores to generate "inverse document frequency"
    topic_idfs = {}
    for t_key in topic_agg_scores.keys():
        topic_idfs[t_key] = math.log(len(e_keys) / (topic_agg_scores[t_key] + 1))

    return ekey_tkey_scores, topic_idfs


def set_episode_topic_tfidf(show_key: ShowKey, topic_key: str, episode_key: str, score: float, topic_idfs: dict, topic_grouping: str, 
                            model_vendor: str, model_version: str):
    episode_topic = esqb.fetch_episode_topic(show_key.value, episode_key, topic_grouping, topic_key, model_vendor, model_version)
    if not episode_topic:
        return
    # use topic.score as a stand-in for "term frequency"
    tfidf_score = score * topic_idfs[topic_key]
    episode_topic.tfidf_score = tfidf_score
    episode_topic.save()
    return episode_topic
