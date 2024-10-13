def flatten_topics(topics: list, parent_only: bool = False, max_rank: int = None):
    '''
    This is the simplest topic flattener and ideally everything should be using it
    '''
    out_list = []
    parents_seen = []

    for i, topic in enumerate(topics):
        if max_rank and i >= max_rank:
            break
        t_bits = topic['topic_key'].split('.')
        if len(t_bits) <= 1 or t_bits[0] in parents_seen:
            continue
        parents_seen.append(t_bits[0])
        if parent_only:
            out_list.append(t_bits[0])
        else:
            out_list.append(topic['topic_key'])

    return ', '.join(out_list)


def flatten_df_list_column(col_contents: list, eligible_col_values: list = None, truncate_at: int = None):
    '''
    Would be a simple list-concat lambda, except that sometimes we want to ignore column values that aren't part of an 'eligible' reference set
    '''
    out_list = []

    i = 0
    for cell_contents in col_contents:
        if truncate_at and i >= truncate_at:
            break
        if eligible_col_values and cell_contents not in eligible_col_values:
            # skip ineligible values without incrementing truncate_at comparison counter
            continue  
        out_list.append(cell_contents)
        i += 1

    return ', '.join(out_list)


def extract_parent(topic_key: str):
    topic_path = topic_key.split('.')
    return topic_path[0]
