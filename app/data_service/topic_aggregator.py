from operator import itemgetter
import pandas as pd


class TopicAgg(object):

    def __init__(self, reference_topics: dict):
        self.reference_topics = reference_topics
        self.keys_to_scores = {}
        self.denominator = 0

    def __repr__(self) -> str:
        return f'keys_to_scores={self.keys_to_scores} denominator={self.denominator}'

    def add_topics(self, topics: list, multiplier: int) -> None:
        for t in topics:
            if t['topic_key'] not in self.keys_to_scores:
                self.keys_to_scores[t['topic_key']] = 0
            self.keys_to_scores[t['topic_key']] += t['dist_score'] * multiplier

        self.denominator += multiplier

    def get_topics(self) -> list:
        sorted_tuples = sorted(self.keys_to_scores.items(), key=itemgetter(1), reverse=True)
        sorted_topics = []
        for st in sorted_tuples:
            if st[0] in self.reference_topics:
                rt_copy = dict(self.reference_topics[st[0]])
                rt_copy['dist_score'] = st[1] / self.denominator
                rt_copy['is_aggregate'] = True
                sorted_topics.append(rt_copy)
            else:
                print(f"TopicAgg.get() warning: topic_key={st[0]} wasn't in reference_topics, not adding to sorted_topic output")
        return sorted_topics
    

def generate_topic_aggs_from_episode_topics(episode_topic_lists: list, max_rank: int = None, max_parent_repeats: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Aggregate topic scores from multiple episodes, output as two dataframes with each topic (parent or leaf) as its own row.
    Takes a list of episode_topics as input. `max_rank` and `max_parent_repeats` inputs for score aggregate tuning.
    '''
    if not max_rank:
        max_rank = 3

    topic_agg_scores = {}
    parent_topic_agg_scores = {}
    topic_agg_tfidf_scores = {}
    parent_topic_agg_tfidf_scores = {}

    for episode_topic_list in episode_topic_lists:
        parents_this_episode = {}
        topic_counter = min(max_rank, len(episode_topic_list))
        for i in range(topic_counter):
            episode_topic = episode_topic_list[i]
            # ignore actual parent topics, do all parent scoring attribution in relation to their child topics
            if episode_topic['is_parent']:
                continue
            topic_key = episode_topic['topic_key']
            parent_topic = topic_key.split('.')[0]
            # aggregate score and ranks for topic
            if topic_key not in topic_agg_scores:
                topic_agg_scores[topic_key] = 0
                topic_agg_tfidf_scores[topic_key] = 0
            topic_agg_scores[topic_key] += episode_topic['score']
            topic_agg_tfidf_scores[topic_key] += episode_topic['tfidf_score']
            # avoid double/triple-scoring using a max_parent_repeats param
            if max_parent_repeats: 
                if parent_topic not in parents_this_episode:
                    parents_this_episode[parent_topic] = 0
                parents_this_episode[parent_topic] += 1
                if parents_this_episode[parent_topic] > max_parent_repeats:
                    continue      
            if parent_topic not in parent_topic_agg_scores:
                parent_topic_agg_scores[parent_topic] = 0
                parent_topic_agg_tfidf_scores[parent_topic] = 0
            parent_topic_agg_scores[parent_topic] += episode_topic['score']
            parent_topic_agg_tfidf_scores[parent_topic] += episode_topic['tfidf_score']

    # topics
    scored_topics = sorted(topic_agg_scores.items(), key=itemgetter(1), reverse=True)
    tfidf_scored_topics = sorted(topic_agg_tfidf_scores.items(), key=itemgetter(1), reverse=True)
    scored_topics_df = pd.DataFrame(scored_topics, columns=['genre', 'score'])
    tfidf_scored_topics_df = pd.DataFrame(tfidf_scored_topics, columns=['genre', 'tfidf_score'])
    topics_df = scored_topics_df.merge(tfidf_scored_topics_df, on='genre')
    topics_df['parent'] = topics_df['genre'].apply(lambda x: x.split('.')[0])
    topics_df.sort_values('parent', inplace=True) # not sure this is needed, or should maybe externalize

    # parent topics
    scored_parent_topics = sorted(parent_topic_agg_scores.items(), key=itemgetter(1), reverse=True)
    tfidf_scored_parent_topics = sorted(parent_topic_agg_tfidf_scores.items(), key=itemgetter(1), reverse=True)
    scored_parent_topics_df = pd.DataFrame(scored_parent_topics, columns=['genre', 'score'])
    tfidf_scored_parent_topics_df = pd.DataFrame(tfidf_scored_parent_topics, columns=['genre', 'tfidf_score'])
    parent_topics_df = scored_parent_topics_df.merge(tfidf_scored_parent_topics_df, on='genre')

    return topics_df, parent_topics_df 
    