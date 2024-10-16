MIN_WORDS_FOR_BERT = 25
MAX_WORDS_FOR_BERT = 500
MIN_SPEAKER_LINES = 5
MIN_SPEAKER_LINE_RATIOS = {2: 0.33, 3: 0.25, 4: 0.2}

BERTOPIC_DATA_DIR = 'bertopic_data'
BERTOPIC_MODELS_DIR = 'bertopic_models'

NTLK_POLARITY = ['pos', 'neg', 'neu']
OPENAI_EMOTIONS = ['Joy', 'Love', 'Empathy', 'Curiosity', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Surprise', 'Confusion']

SENTIMENT_ANALYZERS = ['nltk_pol', 'openai_emo']


WORD2VEC_VENDOR_VERSIONS = {
    # Download webvectors models: http://vectors.nlpl.eu/explore/embeddings/en/models/
    'webvectors': {
        'versions': {
            'gigaword29': {
                'dims': 300,  # 854M
                'status': 'INACTIVE'
            },
            'enwiki223': {
                'dims': 300,  # 691M
                'status': 'ACTIVE'
            }
        },
        'file_suffix': '_model.txt',
        'no_header': False,
        'pos_tag': True
    },
    # Download GloVe models: https://nlp.stanford.edu/projects/glove/
    'glove': {
        'versions': {
            '6B300d': {
                'dims': 300,  # 1.04G
                'status': 'ACTIVE'
            },
            'twitter27B200d': {
                'dims': 200,  # 2.06G
                'status': 'INACTIVE'
            },
            'twitter27B100d': {
                'dims': 100,  # 1.02G
                'status': 'INACTIVE'
            },
            '42B300d': {
                'dims': 300,  # 5.03G
                'status': 'INACTIVE'
            },
            '840B300d': {
                'dims': 300,  # 5.65G
                'status': 'INACTIVE'
            }
        },
        'file_suffix': '_model.txt',
        'no_header': True,
        'pos_tag': False
    },
    # Download fasttext models: https://fasttext.cc/docs/en/english-vectors.html
    'fasttext': {
        'versions': {
            'wikinews300d1M': {
                'dims': 300,  # 2.26G
                'status': 'ACTIVE'
            },
            'crawl300d2M': {
                'dims': 300,  # 4.51G
                'status': 'INACTIVE'
            }
        },
        'file_suffix': '.vec',
        'no_header': False,
        'pos_tag': False
    }
}


TRANSFORMER_VENDOR_VERSIONS = {
    'openai': {
        'versions': {
            'ada002': {
                'dims': 1536,
                'status': 'ACTIVE',
                'true_name': 'text-embedding-ada-002',
                'max_tokens': 8191
            }
        }
    }
}

 
def get_active_models() -> list:
    active_models = []
    for vendor, vendor_meta in WORD2VEC_VENDOR_VERSIONS.items():
        for version, version_meta in vendor_meta['versions'].items():
            if version_meta['status'] == 'ACTIVE':
                active_models.append((vendor, version))
    for vendor, vendor_meta in TRANSFORMER_VENDOR_VERSIONS.items():
        for version, version_meta in vendor_meta['versions'].items():
            if version_meta['status'] == 'ACTIVE':
                # active_models.append((vendor, version_meta['true_name']))
                active_models.append((vendor, version))
    return active_models


ACTIVE_VENDOR_VERSIONS = get_active_models()


# def get_pos_tag_model_vendors() -> list:
#     pos_tag_model_vendors = []
#     for vendor, vendor_meta in WORD2VEC_VENDOR_VERSIONS.items():
#         if vendor_meta['pos_tag']:
#             pos_tag_model_vendors.append(vendor)
#     return pos_tag_model_vendors


# POS_TAG_MODEL_VENDORS = get_pos_tag_model_vendors()
