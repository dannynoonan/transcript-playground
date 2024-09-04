from datetime import datetime
from elasticsearch_dsl import Document, Date, Nested, InnerDoc, Keyword, Text, Integer, analyzer, token_filter, TokenCount, DenseVector, Object, Float, Boolean


freetext_analyzer = analyzer('freetext_analyzer', tokenizer='standard', type='custom',
                             filter=['lowercase', 'stop', 'apostrophe', 'porter_stem'])


token_count_analyzer = analyzer('token_count_analyzer', tokenizer='standard', type='custom')


class EsSceneEvent(InnerDoc):
    context_info = Text(analyzer=freetext_analyzer, term_vector='yes')
    spoken_by = Text(analyzer='standard', fields={'keyword': Keyword()})
    dialog = Text(analyzer=freetext_analyzer, term_vector='yes', fields={'word_count': TokenCount(analyzer=token_count_analyzer, store='true')})
    # generated
    nltk_sent_pos = Float()
    nltk_sent_neg = Float()
    nltk_sent_neu = Float()
    openai_sent_joy = Float()
    openai_sent_love = Float()
    openai_sent_empathy = Float()
    openai_sent_curiosity = Float()
    openai_sent_sadness = Float()
    openai_sent_anger = Float()
    openai_sent_fear = Float()
    openai_sent_disgust = Float()
    openai_sent_surprise = Float()
    openai_sent_confusion = Float()


class EsScene(InnerDoc):
    location = Text(analyzer='standard', fields={'keyword': Keyword()})
    description = Text(analyzer=freetext_analyzer, term_vector='yes')
    scene_events = Nested(EsSceneEvent)
    # generated
    nltk_sent_pos = Float()
    nltk_sent_neg = Float()
    nltk_sent_neu = Float()
    # nltk_speaker_sentiments = Object(multi=True)
    openai_sent_joy = Float()
    openai_sent_love = Float()
    openai_sent_empathy = Float()
    openai_sent_curiosity = Float()
    openai_sent_sadness = Float()
    openai_sent_anger = Float()
    openai_sent_fear = Float()
    openai_sent_disgust = Float()
    openai_sent_surprise = Float()
    openai_sent_confusion = Float()


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
    topics_universal = Object(multi=True)
    topics_focused = Object(multi=True)
    topics_universal_tfidf = Object(multi=True)
    topics_focused_tfidf = Object(multi=True)
    es_mlt_relations_text = Text(multi=True)
    es_mlt_relations_dict = Object(multi=True)
    openai_ada002_relations_text = Text(multi=True)
    openai_ada002_relations_dict = Object(multi=True)
    # webvectors_enwiki223_relations = Text(multi=True)
    # glove_6B300d_relations = Text(multi=True)
    # fasttext_wikinews300d1M_relations = Text(multi=True)
    # embeddings per model
    webvectors_enwiki223_embeddings = DenseVector(dims=300, index='true', similarity='cosine')
    glove_6B300d_embeddings = DenseVector(dims=300, index='true', similarity='cosine')
    fasttext_wikinews300d1M_embeddings = DenseVector(dims=300, index='true', similarity='cosine')
    openai_ada002_embeddings = DenseVector(dims=1536, index='true', similarity='cosine')
    nltk_sent_pos = Float()
    nltk_sent_neg = Float()
    nltk_sent_neu = Float()
    openai_sent_joy = Float()
    openai_sent_love = Float()
    openai_sent_empathy = Float()
    openai_sent_curiosity = Float()
    openai_sent_sadness = Float()
    openai_sent_anger = Float()
    openai_sent_fear = Float()
    openai_sent_disgust = Float()
    openai_sent_surprise = Float()
    openai_sent_confusion = Float()

    class Index:
        name = 'transcripts'

    # def add_scene(self, location, description):
    #     self.scenes.append(
    #       EsScene(location=location, description=description))

    def save(self, **kwargs):
        # self.meta.id = f'{self.show_key}_{self.episode_key}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
    

class EsEpisodeNarrativeSequence(Document):
    show_key = Keyword()
    episode_key = Keyword()
    speaker_group = Keyword()
    narrative_lines = Text(multi=True, analyzer=freetext_analyzer, term_vector='yes', fields={'word_count': TokenCount(analyzer=token_count_analyzer, store='true')})
    word_count = Integer()
    source_scene_word_counts = Object(multi=True)
    speaker_line_counts = Object(multi=True)
    cluster_memberships = Object(multi=True)
    indexed_ts = Date()
    # generated
    nltk_sent_pos = Float()
    nltk_sent_neg = Float()
    nltk_sent_neu = Float()

    class Index:
        name = 'narratives'

    def save(self, **kwargs):
        self.meta.id = f'{self.show_key}_{self.episode_key}_{self.speaker_group}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
    

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
    topics_mbti = Object(multi=True)
    topics_dnda = Object(multi=True)
    openai_ada002_word_count = Integer()
    openai_ada002_embeddings = DenseVector(dims=1536, index='true', similarity='cosine')
    loaded_ts = Date()
    indexed_ts = Date()
    # generated
    nltk_sent_pos = Float()
    nltk_sent_neg = Float()
    nltk_sent_neu = Float()

    class Index:
        name = 'speakers'

    def save(self, **kwargs):
        self.meta.id = f'{self.show_key}_{self.speaker}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
    

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
    topics_mbti = Object(multi=True)
    topics_dnda = Object(multi=True)
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
    topics_mbti = Object(multi=True)
    topics_dnda = Object(multi=True)
    openai_ada002_word_count = Integer()
    openai_ada002_embeddings = DenseVector(dims=1536, index='true', similarity='cosine')
    loaded_ts = Date()
    indexed_ts = Date()
    # generated
    nltk_sent_pos = Float()
    nltk_sent_neg = Float()
    nltk_sent_neu = Float()

    class Index:
        name = 'speaker_episodes'

    def save(self, **kwargs):
        self.meta.id = f'{self.show_key}_{self.speaker}_{self.episode_key}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
    

class EsSpeakerUnified(Document):
    show_key = Keyword()
    speaker = Keyword()
    layer_key = Keyword()
    word_count = Integer()
    openai_ada002_embeddings = DenseVector(dims=1536, index='true', similarity='cosine')
    indexed_ts = Date()

    class Index:
        name = 'speaker_embeddings_unified'

    def save(self, **kwargs):
        self.meta.id = f'{self.show_key}_{self.speaker}_{self.layer_key}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)


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
    

class EsEpisodeTopic(Document):
    show_key = Keyword()
    episode_key = Keyword()
    episode_title = Text()
    season = Integer()
    sequence_in_season = Integer()
    air_date = Date()
    topic_grouping = Keyword()
    topic_key = Keyword()
    topic_name = Keyword()
    is_parent = Boolean()
    raw_score = Float()
    score = Float()  # TODO change to 'dist_score' or 'norm_score'?
    tfidf_score = Float()
    model_vendor = Keyword()
    model_version = Keyword()
    indexed_ts = Date()

    class Index:
        name = 'episode_topics'

    def save(self, **kwargs):
        self.meta.id = f'{self.show_key}_{self.episode_key}_{self.topic_grouping}_{self.topic_key}_{self.model_vendor}_{self.model_version}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)


class EsSpeakerTopic(Document):
    show_key = Keyword()
    speaker = Keyword()
    topic_grouping = Keyword()
    topic_key = Keyword()
    topic_name = Keyword()
    is_parent = Boolean()
    is_aggregate = Boolean()
    raw_score = Float()
    score = Float()
    word_count = Integer()
    model_vendor = Keyword()
    model_version = Keyword()
    indexed_ts = Date()

    class Index:
        name = 'speaker_topics'

    def save(self, **kwargs):
        self.meta.id = f'{self.show_key}_{self.speaker}_{self.topic_grouping}_{self.topic_key}_{self.model_vendor}_{self.model_version}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)


class EsSpeakerSeasonTopic(Document):
    show_key = Keyword()
    speaker = Keyword()
    season = Integer()
    topic_grouping = Keyword()
    topic_key = Keyword()
    topic_name = Keyword()
    is_parent = Boolean()
    is_aggregate = Boolean()
    raw_score = Float()
    score = Float()
    word_count = Integer()
    model_vendor = Keyword()
    model_version = Keyword()
    indexed_ts = Date()

    class Index:
        name = 'speaker_season_topics'

    def save(self, **kwargs):
        self.meta.id = f'{self.show_key}_{self.season}_{self.speaker}_{self.topic_grouping}_{self.topic_key}_{self.model_vendor}_{self.model_version}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)


class EsSpeakerEpisodeTopic(Document):
    show_key = Keyword()
    speaker = Keyword()
    episode_key = Keyword()
    episode_title = Text()
    season = Integer()
    sequence_in_season = Integer()
    air_date = Date()
    topic_grouping = Keyword()
    topic_key = Keyword()
    topic_name = Keyword()
    is_parent = Boolean()
    raw_score = Float()
    score = Float()
    word_count = Integer()
    model_vendor = Keyword()
    model_version = Keyword()
    indexed_ts = Date()

    class Index:
        name = 'speaker_episode_topics'

    def save(self, **kwargs):
        self.meta.id = f'{self.show_key}_{self.episode_key}_{self.speaker}_{self.topic_grouping}_{self.topic_key}_{self.model_vendor}_{self.model_version}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
