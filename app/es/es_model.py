from datetime import datetime
from elasticsearch_dsl import Document, Date, Nested, InnerDoc, Keyword, Text, Integer, analyzer, token_filter, TokenCount, DenseVector, Object, Float


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
    parent_topics = Object(multi=True)
    child_topics = Object(multi=True)
    # topics = Object(multi=True)
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
    
    def set_topics(self, topic_grouping: str, parent_topics: list, child_topics: list) -> None:
        if not self.parent_topics:
            self.parent_topics = {}
        self.parent_topics[topic_grouping] = parent_topics
        if not self.child_topics:
            self.child_topics = {}
        self.child_topics[topic_grouping] = child_topics
    

class EsSpeaker(Document):
    show_key = Keyword()
    speaker = Keyword()
    alt_names = Text(multi=True, analyzer=freetext_analyzer, term_vector='yes')
    actor_names = Text(multi=True, analyzer=freetext_analyzer, term_vector='yes')
    season_count = Integer()
    episode_count = Integer()
    scene_count = Integer()
    line_count = Integer()
    word_count = Integer()
    lines = Text(multi=True, analyzer=freetext_analyzer, term_vector='yes')
    seasons_to_episode_keys = Object(multi=True)
    most_frequent_companions = Object(multi=True)
    parent_topics = Object(multi=True)
    child_topics = Object(multi=True)
    openai_ada002_word_count = Integer()
    openai_ada002_embeddings = DenseVector(dims=1536, index='true', similarity='cosine')
    loaded_ts = Date()
    indexed_ts = Date()

    class Index:
        name = 'speakers'

    def save(self, **kwargs):
        self.meta.id = f'{self.show_key}_{self.speaker}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
    
    def set_topics(self, topic_grouping: str, parent_topics: list, child_topics: list) -> None:
        if not self.parent_topics:
            self.parent_topics = {}
        self.parent_topics[topic_grouping] = parent_topics
        if not self.child_topics:
            self.child_topics = {}
        self.child_topics[topic_grouping] = child_topics
    

class EsSpeakerSeason(Document):
    show_key = Keyword()
    speaker = Keyword()
    season = Integer()
    episode_count = Integer()
    # episode_keys = Keyword(multi=True)
    scene_count = Integer()
    line_count = Integer()
    word_count = Integer()
    lines = Text(multi=True)
    most_frequent_companions = Object(multi=True)
    parent_topics = Object(multi=True)
    child_topics = Object(multi=True)
    openai_ada002_word_count = Integer()
    openai_ada002_embeddings = DenseVector(dims=1536, index='true', similarity='cosine')
    loaded_ts = Date()
    indexed_ts = Date()

    class Index:
        name = 'speaker_seasons'

    def save(self, **kwargs):
        self.meta.id = f'{self.show_key}_{self.speaker}_{self.season}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
    
    def set_topics(self, topic_grouping: str, parent_topics: list, child_topics: list) -> None:
        if not self.parent_topics:
            self.parent_topics = {}
        self.parent_topics[topic_grouping] = parent_topics
        if not self.child_topics:
            self.child_topics = {}
        self.child_topics[topic_grouping] = child_topics
    

class EsSpeakerEpisode(Document):
    show_key = Keyword()
    speaker = Keyword()
    episode_key = Keyword()
    title = Text()
    season = Integer()
    sequence_in_season = Integer()
    air_date = Date()
    agg_score = Float()
    scene_count = Integer()
    line_count = Integer()
    word_count = Integer()
    lines = Text(multi=True)
    most_frequent_companions = Object(multi=True)
    parent_topics = Object(multi=True)
    child_topics = Object(multi=True)
    openai_ada002_word_count = Integer()
    openai_ada002_embeddings = DenseVector(dims=1536, index='true', similarity='cosine')
    loaded_ts = Date()
    indexed_ts = Date()

    class Index:
        name = 'speaker_episodes'

    def save(self, **kwargs):
        self.meta.id = f'{self.show_key}_{self.speaker}_{self.episode_key}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
    
    def set_topics(self, topic_grouping: str, parent_topics: list, child_topics: list) -> None:
        if not self.parent_topics:
            self.parent_topics = {}
        self.parent_topics[topic_grouping] = parent_topics
        if not self.child_topics:
            self.child_topics = {}
        self.child_topics[topic_grouping] = child_topics


class EsTopic(Document):
    topic_grouping = Keyword()
    topic_key = Keyword()
    parent_key = Keyword()
    topic_name = Keyword()
    parent_name = Keyword()
    description = Text()
    parent_description = Text()
    openai_ada002_embeddings = DenseVector(dims=1536, index='true', similarity='cosine')
    indexed_ts = Date()

    class Index:
        name = 'topics'

    def save(self, **kwargs):
        self.meta.id = f'{self.topic_grouping}_{self.topic_key}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
