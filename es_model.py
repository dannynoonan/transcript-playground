from datetime import datetime
from elasticsearch_dsl import Document, Date, Nested, InnerDoc, Keyword, Text, Integer


class EsSceneEvent(InnerDoc):
    context_info = Text()
    spoken_by = Keyword()
    dialog = Text()


class EsScene(InnerDoc):
    location = Keyword()
    description = Text()
    scene_events = Nested(EsSceneEvent)


class EsEpisodeTranscript(Document):
    show_key = Keyword()
    episode_key = Keyword()
    season = Integer()
    sequence_in_season = Integer()
    title = Text()
    air_date = Date()
    duration = Integer()
    scenes = Nested(EsScene)
    loaded_ts = Date()
    indexed_ts = Date()

    class Index:
        name = 'transcripts'

    def save(self, **kwargs):
        # self.meta.id = f'{self.show_key}_{self.episode_key}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)
