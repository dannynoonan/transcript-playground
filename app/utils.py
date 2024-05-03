from operator import itemgetter
import os

from app.es.es_model import EsEpisodeTranscript


def get_or_make_source_dirs(source_type: str, show_key: str = None) -> tuple[str, str]:
    dir_root = 'source'
    if not os.path.isdir(dir_root):
        os.mkdir(dir_root)
    dir_path = f'{dir_root}/{source_type}'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if show_key:
        dir_path = f'{dir_path}/{show_key}'
    backup_dir_path = f'{dir_path}/backup'
    if not os.path.isdir(backup_dir_path):
        os.mkdir(backup_dir_path)

    return dir_path, backup_dir_path


def truncate_dict(d: dict, length: int, start_index: int = 0) -> None:
	end_index = length + start_index
	end_index = min(len(d), end_index)
	return {k: d[k] for k in list(d.keys())[start_index:end_index]}


# def shorten_flattened_text(es_episode: EsEpisodeTranscript, skip_increment: int = None) -> str:
#     flattened_text = f'{es_episode.title} '
#     scene_i = 0
#     for scene in es_episode.scenes:
#         scene_i += 1
#         # if divisor is set, skip scenes at that skip_increment
#         if skip_increment and scene_i % skip_increment == 0:
#             continue
#         # flattened_text += f'{scene.location} '
#         # if scene.description:
#         #     flattened_text += f'{scene.description} '
#         for scene_event in scene.scene_events:
#             # if scene_event.context_info:
#             #     flattened_text += f'{scene_event.context_info} '
#             # if scene_event.spoken_by:
#             #     flattened_text += f'{scene_event.spoken_by}: '
#             if scene_event.dialog:
#                 flattened_text += f'{scene_event.dialog} '
        

#     return flattened_text


def split_parent_and_child_topics(topics: list, parent_limit: int = None, child_limit: int = None) -> tuple[list, list]:
    '''
    output ordered lists of topics (child_topics) and parent_topics
    parent_topics are implied by child_topics and added to respective parent_topics list when child_topics are encountered
    however, parent_topic data is not in scope when parent_key is referenced within child_topic
    parent_keys_to_topics accumulates parent_topic objects (and tracks membership) while parent_keys preserves sequence
    parent_keys_to_topics objects and parent_keys sequence are combined at the end
    '''
    parent_keys_to_topics = {}
    parent_keys = []
    child_topics = []
    for t in topics:
        parent_key = t['parent_key']
        topic_key = t['topic_key']
        # parent topics (which themselves have no parent) are added to parent_topics if not already present in parent_keys_added
        if not parent_key:
            if topic_key not in parent_keys_to_topics:
                parent_keys.append(topic_key)
            parent_keys_to_topics[topic_key] = t # NOTE weird way to redundantly backfill topic data for parent keys as we go 
            # elif parent_keys_to_topics[topic_key] is None:
            #     parent_keys_to_topics[topic_key] = t
        # child topics also add their parent topics (if not already present in parent_keys_added)
        else:
            if parent_key not in parent_keys_to_topics:
                parent_keys.append(parent_key)
                parent_keys_to_topics[parent_key] = None
            if topic_key not in child_topics:
                child_topics.append(t)
    # combine sequenced parent_keys with objects stored in parent_keys_to_topics to return sequenced parent_topics
    parent_topics = []
    for pk in parent_keys:
        if pk in parent_keys_to_topics:
            parent_topics.append(parent_keys_to_topics[pk])

    if parent_limit and len(parent_topics) > parent_limit:
        parent_topics = parent_topics[:parent_limit]
    if child_limit and len(child_topics) > child_limit:
        child_topics = child_topics[:child_limit]

    return parent_topics, child_topics


# def update_topic_agg(topic_aggs: dict, topics: list, multiplier: int) -> None:
#     for t in topics:
#         if t['topic_key'] not in topic_aggs:
#             topic_aggs[t['topic_key']] = 0
#         topic_aggs[t['topic_key']] += t['score'] * multiplier


# def sort_topic_aggs(topic_aggs: dict, reference_topics: dict) -> list:
#     sorted_topic_tuples = sorted(topic_aggs.items(), key=itemgetter(1), reverse=True)
#     sorted_topics = []
#     rank = 1
#     for tt in sorted_topic_tuples:
#         if tt[0] in reference_topics:
#             rt = reference_topics[tt[0]]
#             rt['rank'] = rank
#             rt['agg_score'] = tt[1]
#             sorted_topics.append(reference_topics[tt[0]])
#             rank += 1
#         else:
#             print(f"in sort_topic_aggs: topic_aggs topic_key={tt[0]} wasn't found in reference_topics, so not adding to sorted_topic output")
#     return sorted_topics


class TopicAgg(object):

    def __init__(self, reference_topics: list):
        self.reference_topics = reference_topics
        self.parent_topic_scores = {}
        self.child_topic_scores = {}
        self.denominator = 0

    def __repr__(self) -> str:
        return f'parent_topic_scores={self.parent_topic_scores} child_topic_scores={self.child_topic_scores} denominator={self.denominator}'
    
    def update(self, parent_topics: list, child_topics: list, multiplier: int) -> None:
        for t in parent_topics:
            if t['topic_key'] not in self.parent_topic_scores:
                self.parent_topic_scores[t['topic_key']] = 0
            self.parent_topic_scores[t['topic_key']] += t['score'] * multiplier
        
        for t in child_topics:
            if t['topic_key'] not in self.child_topic_scores:
                self.child_topic_scores[t['topic_key']] = 0
            self.child_topic_scores[t['topic_key']] += t['score'] * multiplier

        self.denominator += multiplier

    def return_sorted(self, parent_limit: int = None, child_limit: int = None) -> tuple[list, list]:

        sorted_parent_tuples = sorted(self.parent_topic_scores.items(), key=itemgetter(1), reverse=True)
        sorted_parent_topics = []
        rank = 1
        for st in sorted_parent_tuples:
            if st[0] in self.reference_topics:
                rt = self.reference_topics[st[0]]
                rt['rank'] = rank
                rt['score'] = st[1] / self.denominator
                sorted_parent_topics.append(self.reference_topics[st[0]])
                rank += 1
            else:
                print(f"TopicAgg.get() warning: topic_key={st[0]} wasn't in reference_topics, not adding to sorted_topic output")

        sorted_child_tuples = sorted(self.child_topic_scores.items(), key=itemgetter(1), reverse=True)
        sorted_child_topics = []
        rank = 1
        for st in sorted_child_tuples:
            if st[0] in self.reference_topics:
                rt = self.reference_topics[st[0]]
                rt['rank'] = rank
                rt['score'] = st[1] / self.denominator
                sorted_child_topics.append(self.reference_topics[st[0]])
                rank += 1
            else:
                print(f"TopicAgg.get() warning: topic_key={st[0]} wasn't in reference_topics, not adding to sorted_topic output")

        if parent_limit and len(sorted_parent_topics) > parent_limit:
            sorted_parent_topics = sorted_parent_topics[:parent_limit]
        if child_limit and len(sorted_child_topics) > child_limit:
            sorted_child_topics = sorted_child_topics[:child_limit]

        return sorted_parent_topics, sorted_child_topics
