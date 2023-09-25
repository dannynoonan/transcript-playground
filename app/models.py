from tortoise.models import Model
from tortoise import fields

'''
https://medium.com/@talhakhalid101/python-tortoise-orm-integration-with-fastapi-c3751d248ce1
https://tortoise.github.io/contrib/fastapi.html#tortoise.contrib.fastapi.register_tortoise.app
'''

class Job(Model):
    # Primary key field is created automatically
    # id = fields.IntField(pk=True) 
    name = fields.CharField(max_length=255)
    description = fields.TextField()

    def __str__(self):
        return self.name
    

class Show(Model):
    key = fields.CharField(max_length=255, unique=True)
    title = fields.TextField()

    episodes: fields.ReverseRelation["Episode"]
    # locations: fields.ReverseRelation["Location"]
    # characters: fields.ReverseRelation["Character"]
    
    def __str__(self):
        return self.key


class RawEpisode(Model):
    show_key = fields.CharField(max_length=255)
    transcript_type = fields.CharField(max_length=255)
    transcript_url = fields.CharField(max_length=1024, unique=True)
    external_key = fields.CharField(max_length=255)
    loaded_ts = fields.DatetimeField(auto_now=True)

    class Meta:
        unique_together=(("show_key", "external_key"))
        table=("raw_episode")

    def __str__(self):
        return str(f'{self.show_key}:{self.external_key}')

    def __repr__(self):
        return str(self)


# class Location(Model):
#     show: fields.ForeignKeyRelation[Show] = fields.ForeignKeyField('models.Show', related_name='locations')
#     name = fields.CharField(max_length=255)

#     class Meta:
#         unique_together=(("show", "name"))

#     def __str__(self):
#         return self.name
    
#     def __repr__(self):
#         return str(self)
    

# class Character(Model):
#     show: fields.ForeignKeyRelation[Show] = fields.ForeignKeyField('models.Show', related_name='characters')
#     name = fields.CharField(max_length=255)

#     class Meta:
#         unique_together=(("show", "name"))

#     def __str__(self):
#         return self.name
    
#     def __repr__(self):
#         return str(self)
    
    
class Episode(Model):
    show: fields.ForeignKeyRelation[Show] = fields.ForeignKeyField('models.Show', related_name='episodes')
    # show = fields.ForeignKeyField('models.Show', related_name='episodes')
    season = fields.IntField()
    sequence_in_season = fields.IntField()
    external_key = fields.CharField(max_length=255, unique=True)
    title = fields.TextField()
    # transcript_url = fields.CharField(max_length=1023, unique=True)
    air_date = fields.DateField(null=True)
    duration = fields.FloatField(null=True)
    loaded_ts = fields.DatetimeField(auto_now=True)

    scenes: fields.ReverseRelation["Scene"]

    class Meta:
        unique_together=(("show", "season", "sequence_in_season"))

    def __str__(self):
        return str(f'{self.show}:{self.external_key}')
    
    def __repr__(self):
        return str(self)


class Scene(Model):
    episode: fields.ForeignKeyRelation[Episode] = fields.ForeignKeyField('models.Episode', related_name='scenes')
    # episode = fields.ForeignKeyField('models.Episode', related_name='scenes')
    sequence_in_episode = fields.IntField()
    # location = fields.ManyToManyField('models.Location', related_name='scenes', through='scene_location')
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
    # dialogue_spoken_by = fields.ForeignKeyField('models.Character', related_name='event')
    dialogue_spoken_by = fields.CharField(max_length=255, null=True)
    dialogue_text = fields.TextField(null=True)

    class Meta:
        unique_together=(("scene", "sequence_in_scene"))
        table=("scene_event")

    def __str__(self):
        event_meta = ''
        if self.context_info:
            event_meta = f'{self.context_info}:'
        if self.dialogue_spoken_by:
            event_meta += f'{self.dialogue_spoken_by}:{self.dialogue_text}'
        return str(f'{self.scene}:{self.sequence_in_scene}:{event_meta}')
    
    def __repr__(self):
        return str(self)
