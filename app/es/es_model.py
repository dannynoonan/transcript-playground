from datetime import datetime
from elasticsearch_dsl import Document, Date, Nested, InnerDoc, Keyword, Text, Integer, analyzer, token_filter, TokenCount, DenseVector, Object


freetext_analyzer = analyzer('freetext_analyzer', tokenizer='standard', type='custom',
                             filter=['lowercase', 'stop', 'apostrophe', 'porter_stem'])


token_count_analyzer = analyzer('token_count_analyzer', tokenizer='standard', type='custom')


class EsSceneEvent(InnerDoc):
    context_info = Text(analyzer=freetext_analyzer, term_vector='yes')
    spoken_by = Text(analyzer='standard', fields={'keyword': Keyword()})
    dialog = Text(analyzer=freetext_analyzer, term_vector='yes', fields={'word_count': TokenCount(analyzer=token_count_analyzer, store='true')})
    # generated
    # cbow_embedding = DenseVector(dims=100, index='true', similarity='cosine')
    # skipgram_embedding = DenseVector(dims=100, index='true', similarity='cosine')


class EsScene(InnerDoc):
    location = Text(analyzer='standard', fields={'keyword': Keyword()})
    description = Text(analyzer=freetext_analyzer, term_vector='yes')
    scene_events = Nested(EsSceneEvent)
    # generated
    # cbow_embedding = DenseVector(dims=100, index='true', similarity='cosine')
    # skipgram_embedding = DenseVector(dims=100, index='true', similarity='cosine')


class EsEpisodeTranscript(Document):
    show_key = Keyword()
    episode_key = Keyword()
    season = Integer()
    sequence_in_season = Integer()
    title = Text(analyzer=freetext_analyzer, term_vector='yes', fields={'word_count': TokenCount(analyzer=token_count_analyzer, store='true')})
    air_date = Date()
    duration = Integer()
    scenes = Nested(EsScene)
    scene_count = Integer()
    loaded_ts = Date()
    indexed_ts = Date()
    flattened_text = Text(analyzer=freetext_analyzer, term_vector='yes', fields={'word_count': TokenCount(analyzer=token_count_analyzer, store='true')})
    # generated 
    focal_speakers = Keyword(multi=True)
    focal_locations = Keyword(multi=True)
    es_mlt_relations_text = Text(multi=True)
    openai_ada002_relations_text = Text(multi=True)
    es_mlt_relations_tuple = Object(multi=True)
    openai_ada002_relations_tuple = Object(multi=True)
    es_mlt_relations_dict = Object(multi=True)
    openai_ada002_relations_dict = Object(multi=True)
    # webvectors_gigaword29_relations = Text(multi=True)
    # webvectors_enwiki223_relations = Text(multi=True)
    # glove_6B300d_relations = Text(multi=True)
    # glove_twitter27B200d_relations = Text(multi=True)
    # glove_twitter27B100d_relations = Text(multi=True)
    # glove_42B300d_relations = Text(multi=True)
    # glove_840B300d_relations = Text(multi=True)
    # fasttext_wikinews300d1M_relations = Text(multi=True)
    # fasttext_crawl300d2M_relations = Text(multi=True)
    # embeddings per model
    webvectors_gigaword29_embeddings = DenseVector(dims=300, index='true', similarity='cosine')
    webvectors_enwiki223_embeddings = DenseVector(dims=300, index='true', similarity='cosine')
    glove_6B300d_embeddings = DenseVector(dims=300, index='true', similarity='cosine')
    glove_twitter27B200d_embeddings = DenseVector(dims=200, index='true', similarity='cosine')
    glove_twitter27B100d_embeddings = DenseVector(dims=100, index='true', similarity='cosine')
    glove_42B300d_embeddings = DenseVector(dims=300, index='true', similarity='cosine')
    glove_840B300d_embeddings = DenseVector(dims=300, index='true', similarity='cosine')
    fasttext_wikinews300d1M_embeddings = DenseVector(dims=300, index='true', similarity='cosine')
    fasttext_crawl300d2M_embeddings = DenseVector(dims=300, index='true', similarity='cosine')
    openai_ada002_embeddings = DenseVector(dims=1536, index='true', similarity='cosine')
    # matched tokens per model
    webvectors_gigaword29_tokens = Text()
    webvectors_enwiki223_tokens = Text()
    glove_6B300d_tokens = Text()
    glove_twitter27B200d_tokens = Text()
    glove_twitter27B100d_tokens = Text()
    glove_42B300d_tokens = Text()
    glove_840B300d_tokens = Text()
    fasttext_wikinews300d1M_tokens = Text()
    fasttext_crawl300d2M_tokens = Text()
    # unmatched tokens per model
    webvectors_gigaword29_no_match_tokens = Text()
    webvectors_enwiki223_no_match_tokens = Text()
    glove_6B300d_no_match_tokens = Text()
    glove_twitter27B200d_no_match_tokens = Text()
    glove_twitter27B100d_no_match_tokens = Text()
    glove_42B300d_no_match_tokens = Text()
    glove_840B300d_no_match_tokens = Text()
    fasttext_wikinews300d1M_no_match_tokens = Text()
    fasttext_crawl300d2M_no_match_tokens = Text()


    class Index:
        name = 'transcripts'

    # def add_scene(self, location, description):
    #     self.scenes.append(
    #       EsScene(location=location, description=description))

    def save(self, **kwargs):
        # self.meta.id = f'{self.show_key}_{self.episode_key}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
    

class Character(Document):
    show_key = Keyword()
    alt_names = Keyword(multi=True)
    actor_names = Keyword(multi=True)
    season_count = Integer()
    scene_count = Integer()
    scene_event_count = Integer()
    word_count = Integer()
    most_frequent_companions = Keyword(multi=True)
    episode_embeddings = DenseVector(dims=1536, index='true', similarity='cosine')
    # episode_classifications 
    # season_classifications
    # show_classification
