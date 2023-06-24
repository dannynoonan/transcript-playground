import json


class Jsonable(object):
    def __init__(self):
        pass

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            ensure_ascii=False, sort_keys=True, indent=4)


class Show(Jsonable):
    def __init__(self):
        super().__init__()
        self.name = None


class Character(Jsonable):
    def __init__(self):
        super().__init__()
        self.name = None
        self.gender = None
        self.species = None


class Location(Jsonable):
    def __init__(self):
        self.name = None


class Episode(Jsonable):
    def __init__(self):
        super().__init__()
        self.show = None  # Show
        self.season = None  # int
        self.sequence_in_season = None  # int
        self.title = None  # str
        self.transcript_url = None  # str
        self.air_date = None  # date
        self.duration = None  # int
        self.scenes = []  # Scene[]


class Scene(Jsonable):
    def __init__(self):
        super().__init__()
        self.location = None  # Location
        self.description = None  # str
        self.events = []  # SceneEvent[]


class SceneEvent(Jsonable):
    def __init__(self):
        super().__init__()
        self.context_info = None  # str
        self.characters_mentioned = []  # Character[]


class SceneDialog(SceneEvent):
    def __init__(self):
        super().__init__()
        self.spoken_by = None  # Character
        self.line = None  # str
