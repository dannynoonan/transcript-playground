import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import re

from show_metadata import build_query_replacement_map, build_query_supplement_map, build_query_expansion_map


nltk.download('stopwords') # run this command to download the stopwords in the project
nltk.download('punkt') # essential for tokenization


query_replacement_map = {}
query_supplement_map = {}
query_expansion_map = {}


def standardize_and_tokenize_query(text: str, tag_pos: bool = False) -> str:
    # tokenize
    tokens = word_tokenize(text)
    # remove stopwords
    tokens = [w.lower().strip() for w in tokens if not w.lower() in stopwords.words("english")]
    # pos tag
    if tag_pos:
        pos_tokens = pos_tag(tokens, tagset='universal')
        for i in range(len(pos_tokens)):
            tokens[i] = f'{pos_tokens[i][0]}_{pos_tokens[i][1]}'

    # print(f'tokens={tokens}')
    return tokens


def normalize_and_expand_query(qt: str, show_key: str) -> str:
    # print(f'qt before normalize_query and expand_query={qt}')
    qt = qt.lower()
    # remove numbers and special characters
    qt = re.sub("[^A-Za-z]+", " ", qt)
    # bookend qt with spaces so we can scan each term flanked by spaces without missing first and last
    qt = f' {qt} '  
    # normalize query using ontological metadata
    qt = normalize_query(qt, show_key)
    # expand query using ontological metadata
    qt = expand_query(qt, show_key)
    return qt


def normalize_query(qt: str, show_key: str) -> str:
    # replace common mis-spellings and other invalid terms with proper replacements
    if show_key not in query_replacement_map:
        query_replacement_map[show_key] = build_query_replacement_map(show_key)
    for term, repl_terms in query_replacement_map[show_key].items():
        if f' {term} ' in qt:
            qt = qt.replace(term, " ".join(repl_terms))
    # supplement terms with alternative variants that increase query match potential
    if show_key not in query_supplement_map:
        query_supplement_map[show_key] = build_query_supplement_map(show_key)
    for term, supp_terms in query_supplement_map[show_key].items():
        if f' {term} ' in qt:
            refined_supp_terms = []
            for supp_term in supp_terms:
                if f' {supp_term} ' not in qt:
                    refined_supp_terms.append(supp_term)
            if refined_supp_terms:
                qt = qt.replace(term, f'{" ".join(refined_supp_terms)} {term}')
    return qt


def expand_query(qt: str, show_key: str) -> str:
    # supplement terms with alternative variants that increase query match potential
    if show_key not in query_expansion_map:
        query_expansion_map[show_key] = build_query_expansion_map(show_key)
    for term, exp_terms in query_expansion_map[show_key].items():
        if f' {term} ' in qt:
            refined_exp_terms = []
            for exp_term in exp_terms:
                if f' {exp_term} ' not in qt:
                    refined_exp_terms.append(exp_term)
            qt = qt.replace(term, f'{" ".join(refined_exp_terms)} {term}')
    return qt