from operator import itemgetter
import os


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


def merge_sorted_lists(list1: list, list2: list) -> list:
    keywords_to_ranks = {}

    for i, kw in list(enumerate(list1)):
        if kw not in keywords_to_ranks:
            keywords_to_ranks[kw] = 0
        keywords_to_ranks[kw] += len(list1)-i
    for i, kw in list(enumerate(list2)):
        if kw not in keywords_to_ranks:
            keywords_to_ranks[kw] = 0
        keywords_to_ranks[kw] += len(list2)-i
    
    merged_sorted_list = [k for k, _ in sorted(keywords_to_ranks.items(), key=lambda kv: kv[1], reverse=True)]
    return merged_sorted_list


def truncate(text: str) -> str:
    if len(text) > 80:
        text = f'{text[:80]}...'
    return text


def wrap_title_in_url(show_key: str, episode_key: str) -> str:
    return f'[link](/web/episode/{show_key}/{episode_key})'


# @DeprecationWarning
# def split_parent_and_child_topics(topics: list, parent_limit: int = None, child_limit: int = None) -> tuple[list, list]:
#     '''
#     output ordered lists of topics (child_topics) and parent_topics
#     parent_topics are implied by child_topics and added to respective parent_topics list when child_topics are encountered
#     however, parent_topic data is not in scope when parent_key is referenced within child_topic
#     parent_keys_to_topics accumulates parent_topic objects (and tracks membership) while parent_keys preserves sequence
#     parent_keys_to_topics objects and parent_keys sequence are combined at the end
#     '''
#     parent_keys_to_topics = {}
#     parent_keys = []
#     child_topics = []
#     for t in topics:
#         parent_key = t['parent_key']
#         topic_key = t['topic_key']
#         # parent topics (which themselves have no parent) are added to parent_topics if not already present in parent_keys_added
#         if not parent_key:
#             if topic_key not in parent_keys_to_topics:
#                 parent_keys.append(topic_key)
#             parent_keys_to_topics[topic_key] = t # NOTE weird way to redundantly backfill topic data for parent keys as we go 
#             # elif parent_keys_to_topics[topic_key] is None:
#             #     parent_keys_to_topics[topic_key] = t
#         # child topics also add their parent topics (if not already present in parent_keys_added)
#         else:
#             if parent_key not in parent_keys_to_topics:
#                 parent_keys.append(parent_key)
#                 parent_keys_to_topics[parent_key] = None
#             if topic_key not in child_topics:
#                 child_topics.append(t)
#     # combine sequenced parent_keys with objects stored in parent_keys_to_topics to return sequenced parent_topics
#     parent_topics = []
#     for pk in parent_keys:
#         if pk in parent_keys_to_topics:
#             parent_topics.append(parent_keys_to_topics[pk])

#     if parent_limit and len(parent_topics) > parent_limit:
#         parent_topics = parent_topics[:parent_limit]
#     if child_limit and len(child_topics) > child_limit:
#         child_topics = child_topics[:child_limit]

#     return parent_topics, child_topics


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
    
    
def flatten_topics(topics: list) -> list:
    simple_topics = []
    count = 0
    for t in topics:
        if 'is_parent' in t and t['is_parent']:
            continue
        simple_topic = dict(topic_key=t['topic_key'], topic_name=t['topic_name'], score=t['score'], raw_score=t['raw_score'])
        if 'tfidf_score' in t:
            simple_topic['tfidf_score'] = t['tfidf_score']
        simple_topics.append(simple_topic)
        count += 1
        if count > 5:
            break
    return simple_topics
