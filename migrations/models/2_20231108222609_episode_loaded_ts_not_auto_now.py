from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "episode" ALTER COLUMN "loaded_ts" TYPE TIMESTAMPTZ USING "loaded_ts"::TIMESTAMPTZ;
        ALTER TABLE "episode" ALTER COLUMN "loaded_ts" DROP NOT NULL;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "episode" ALTER COLUMN "loaded_ts" SET NOT NULL;
        ALTER TABLE "episode" ALTER COLUMN "loaded_ts" TYPE TIMESTAMPTZ USING "loaded_ts"::TIMESTAMPTZ;"""
