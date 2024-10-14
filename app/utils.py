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


def set_dict_value_as_es_value(es_object: object, d: dict, k: str, es_field_prefix: str) -> None:
    if not d or k not in d:
        print(f'failed to set_dict_value_as_es_value, k={k} is not in d={d}')
        return
    es_field = f'{es_field_prefix}{k.lower()}'
    es_object[es_field] = d[k]


def extract_phrase_qts(qt: str) -> tuple[str, list]:
    if not qt:
        return None, []
    
    # remove internal whitespace
    qt = ' '.join(qt.split())
    # if no quotes present, there's no phrase to extract
    if not '"' in qt:
        return qt, []
    # carve up qt into portions between quotes and outside of quotes
    tokens = []
    phrases = []
    qt_bits = qt.split('"')
    inside_quotes = False
    # NOTE we're not verifying an even number of quotes; text after an odd quote is always treated as a phrase until the next even quote
    for bit in qt_bits:
        if inside_quotes:
            phrases.append(bit.strip())
            inside_quotes = False
        else:
            if bit:
                bit = bit.strip()
                if bit:
                    tokens.append(bit)
            inside_quotes = True

    return ' '.join(tokens), phrases


def hilite_in_logs(message: object) -> None:
    print('****************************************************************************************************************')
    print('****************************************************************************************************************')
    print(message)
    print('****************************************************************************************************************')
    print('****************************************************************************************************************')
