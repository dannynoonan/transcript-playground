from datetime import datetime
from elasticsearch_dsl import Document, Date, Nested, InnerDoc, Keyword, Text, Integer, analyzer, token_filter, TokenCount


freetext_analyzer = analyzer('freetext_analyzer', tokenizer='standard', type='custom',
                             filter=['lowercase', 'stop', 'apostrophe', 'porter_stem'])


token_count_analyzer = analyzer('token_count_analyzer', tokenizer='standard', type='custom')


class EsSceneEvent(InnerDoc):
    context_info = Text(analyzer=freetext_analyzer, term_vector='yes')
    spoken_by = Text(analyzer='standard', fields={'keyword': Keyword()})
    dialog = Text(analyzer=freetext_analyzer, term_vector='yes', fields={'word_count': TokenCount(analyzer=token_count_analyzer, store='true')})


class EsScene(InnerDoc):
    location = Text(analyzer='standard', fields={'keyword': Keyword()})
    description = Text(analyzer=freetext_analyzer, term_vector='yes')
    scene_events = Nested(EsSceneEvent)


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

    class Index:
        name = 'transcripts'

    # def add_scene(self, location, description):
    #     self.scenes.append(
    #       EsScene(location=location, description=description))

    def save(self, **kwargs):
        # self.meta.id = f'{self.show_key}_{self.episode_key}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
