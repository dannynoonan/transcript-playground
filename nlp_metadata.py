WORD2VEC_VENDOR_VERSIONS = {
    'webvectors': {
        'versions': {
            'gigaword29': {
                'dims': 300,  # 854M
                'status': 'ACTIVE'
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
    'glove': {
        'versions': {
            '6B300d': {
                'dims': 300,  # 1.04G
                'status': 'ACTIVE'
            },
            'twitter27B200d': {
                'dims': 200,  # 2.06G
                'status': 'ACTIVE'
            },
            'twitter27B100d': {
                'dims': 100,  # 1.02G
                'status': 'ACTIVE'
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

 
def get_active_models() -> dict:
    active_models = {}
    for vendor, vendor_meta in WORD2VEC_VENDOR_VERSIONS.items():
        active_versions = []
        for version, version_meta in vendor_meta['versions'].items():
            if version_meta['status'] == 'ACTIVE':
                active_versions.append(version)
        active_models[vendor] = active_versions
    return active_models


ACTIVE_VENDOR_VERSIONS = get_active_models()


# def get_pos_tag_model_vendors() -> list:
#     pos_tag_model_vendors = []
#     for vendor, vendor_meta in WORD2VEC_VENDOR_VERSIONS.items():
#         if vendor_meta['pos_tag']:
#             pos_tag_model_vendors.append(vendor)
#     return pos_tag_model_vendors


# POS_TAG_MODEL_VENDORS = get_pos_tag_model_vendors()
