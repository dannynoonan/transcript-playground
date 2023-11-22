from datetime import datetime
from elasticsearch_dsl import Document, Date, Nested, InnerDoc, Keyword, Text, Integer, Long


class EsSceneEvent(InnerDoc):
    context_info = Text()
    spoken_by = Keyword()
    # spoken_by = Text(fields={'keyword': Keyword()})
    dialog = Text()


class EsScene(InnerDoc):
    location = Keyword()
    # location = Text(fields={'keyword': Keyword()})
    description = Text()
    scene_events = Nested(EsSceneEvent)


class EsEpisodeTranscript(Document):
    show_key = Keyword()
    # show_key = Text(fields={'keyword': Keyword()})
    episode_key = Keyword()
    # episode_key = Text(fields={'keyword': Keyword()})
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

    # def add_scene(self, location, description):
    #     self.scenes.append(
    #       EsScene(location=location, description=description))

    def save(self, **kwargs):
        # self.meta.id = f'{self.show_key}_{self.episode_key}'
        self.indexed_ts = datetime.now()
        return super().save(**kwargs)


class ScoreMeta(object):
    # score = Long()
    # agg_score = Long()
    # high_child_score = Long()

    def __init__(self, score: Long = 0, agg_score: Long = 0, high_child_score: Long = 0):
        self.score = score
        self.agg_score = agg_score
        self.high_child_score = high_child_score

class SortableDict(object):
    def __init__(self, obj: dict):
        self.obj = obj

    def __gt__(self, other):
        if self.obj['score_meta'].agg_score > other.obj['score_meta'].agg_score:
            return True
        return False
