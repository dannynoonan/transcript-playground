from tortoise.models import Model
from tortoise import fields

'''
https://medium.com/@talhakhalid101/python-tortoise-orm-integration-with-fastapi-c3751d248ce1
https://tortoise.github.io/contrib/fastapi.html#tortoise.contrib.fastapi.register_tortoise.app
''' 

class Episode(Model):
    show_key = fields.CharField(max_length=255)
    season = fields.IntField()
    sequence_in_season = fields.IntField()
    external_key = fields.CharField(max_length=255)
    title = fields.TextField()
    air_date = fields.DateField(null=True)
    duration = fields.FloatField(null=True)
    loaded_ts = fields.DatetimeField(null=True)
    transcript_loaded_ts = fields.DatetimeField(null=True)

    scenes: fields.ReverseRelation["Scene"]
    transcript_sources: fields.ReverseRelation["TranscriptSource"]

    class Meta:
        unique_together=(("show_key", "season", "sequence_in_season"))
        unique_together=(("show_key", "external_key"))

    def __str__(self):
        return str(f'{self.show_key}:{self.external_key}')
    
    def __repr__(self):
        return str(self)


class TranscriptSource(Model):
    episode: fields.ForeignKeyRelation[Episode] = fields.ForeignKeyField('models.Episode', related_name='transcript_sources')
    transcript_type = fields.CharField(max_length=255)
    transcript_url = fields.CharField(max_length=1024, unique=True)
    loaded_ts = fields.DatetimeField(auto_now=True)

    class Meta:
        unique_together=(("episode", "transcript_type"))
        table=("transcript_source")

    def __str__(self):
        return str(f'{self.episode}:{self.transcript_type}')

    def __repr__(self):
        return str(self)
    

class Scene(Model):
    episode: fields.ForeignKeyRelation[Episode] = fields.ForeignKeyField('models.Episode', related_name='scenes')
    sequence_in_episode = fields.IntField()
    location = fields.CharField(max_length=255)
    description = fields.TextField(null=True)

    events: fields.ReverseRelation["SceneEvent"]

    class Meta:
        unique_together=(("episode", "sequence_in_episode"))

    def __str__(self):
        return str(f'{self.episode}:{self.sequence_in_episode}:{self.location}')
    
    def __repr__(self):
        return str(self)


class SceneEvent(Model):
    scene: fields.ForeignKeyRelation[Scene] = fields.ForeignKeyField('models.Scene', related_name='events')
    sequence_in_scene = fields.IntField()
    context_info = fields.TextField(null=True)
    dialogue_spoken_by = fields.CharField(max_length=255, null=True)
    dialogue_text = fields.TextField(null=True)

    class Meta:
        unique_together=(("scene", "sequence_in_scene"))
        table=("scene_event")

    def __str__(self):
        # event_meta = ''
        # if self.context_info:
        #     event_meta = f'{self.context_info}:'
        # if self.dialogue_spoken_by:
        #     event_meta += f'{self.dialogue_spoken_by}:{self.dialogue_text}'
        # return str(f'{self.scene}:{self.sequence_in_scene}:{event_meta}')
        return str(f'{self.scene}:{self.sequence_in_scene}')
    
    def __repr__(self):
        return str(self)
