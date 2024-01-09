from nlp_metadata import ACTIVE_VENDOR_VERSIONS

STOPWORDS = ["a", "able", "about", "across", "after", "again", "all", "almost", "also", "am", "among", "an", "and", "ani", "anoth", "any", "anyth", 
                "are", "as", "at", 
            "back", "be", "because", "been", "but", "by", 
            "c", "can", "cannot", "could", 
            "dear", "did", "didn", "do", "does", "don",
            "either", "else", "ever", "every", 
            "for", "from", 
            "get", "go", "got", 
            "had", "has", "have", "ha", "he", "her", "here", "hers", "hi", "him", "his", "how", "however", 
            "i", "if", "in", "into", "is", "it", "its", 
            "just", 
            "least", "leav", "let", "like", "likely", 
            "make", "may", "me", "might", "more", "most", "must", "my", 
            "neither", "no", "nor", "not", "noth", "now",
            "o", "oc", "of", "off", "often", "oh", "on", "onli", "only", "or", "other", "our", "out", "own", 
            "rather", 
            "said", "say", "says", "see", "she", "should", "since", "so", "some", 
            "take", "than", "that", "the", "thei", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", 
            "up", "us", 
            "veri",
            "wai", "wants", "was", "we", "well", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", 
            "ye", "yet", "you", "your",

            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twenty",
            "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            
            "barclai", "la", "forg", "geordi", "laforg", "weslei"]


# HOMEGROWN_MODEL_TYPES = ['cbow', 'sg']


# ACTIVE_VENDOR_MODELS = {
#     'webvectors': ['gigaword29', 'enwiki223'],
#     'glove': ['6B300d', 'twitter27B200d', 'twitter27B100d'],
#     'fasttext': ['wikinews300d1M']
# }


def concat_vector_fields() -> list:
    vector_fields = []
    for vendor_version in ACTIVE_VENDOR_VERSIONS:
        vector_fields.append(f'{vendor_version[0]}_{vendor_version[1]}_embeddings')
        vector_fields.append(f'{vendor_version[0]}_{vendor_version[1]}_tokens')
        vector_fields.append(f'{vendor_version[0]}_{vendor_version[1]}_no_match_tokens')
    return vector_fields


VECTOR_FIELDS = concat_vector_fields()
