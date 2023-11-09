from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "episode" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "show_key" VARCHAR(255) NOT NULL,
    "season" INT NOT NULL,
    "sequence_in_season" INT NOT NULL,
    "external_key" VARCHAR(255) NOT NULL,
    "title" TEXT NOT NULL,
    "air_date" DATE,
    "duration" DOUBLE PRECISION,
    "loaded_ts" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "uid_episode_show_ke_a62e7b" UNIQUE ("show_key", "external_key")
);
CREATE TABLE IF NOT EXISTS "scene" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "sequence_in_episode" INT NOT NULL,
    "location" VARCHAR(255) NOT NULL,
    "description" TEXT,
    "episode_id" INT NOT NULL REFERENCES "episode" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_scene_episode_1c3fd1" UNIQUE ("episode_id", "sequence_in_episode")
);
CREATE TABLE IF NOT EXISTS "scene_event" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "sequence_in_scene" INT NOT NULL,
    "context_info" TEXT,
    "dialogue_spoken_by" VARCHAR(255),
    "dialogue_text" TEXT,
    "scene_id" INT NOT NULL REFERENCES "scene" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_scene_event_scene_i_dff350" UNIQUE ("scene_id", "sequence_in_scene")
);
CREATE TABLE IF NOT EXISTS "transcript_source" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "transcript_type" VARCHAR(255) NOT NULL,
    "transcript_url" VARCHAR(1024) NOT NULL UNIQUE,
    "loaded_ts" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "episode_id" INT NOT NULL REFERENCES "episode" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_transcript__episode_1dcdd2" UNIQUE ("episode_id", "transcript_type")
);
CREATE TABLE IF NOT EXISTS "aerich" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "version" VARCHAR(255) NOT NULL,
    "app" VARCHAR(100) NOT NULL,
    "content" JSONB NOT NULL
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
