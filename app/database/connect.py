from tortoise import Tortoise

from app.config import DATABASE_URL


async def connect_to_database():
    await Tortoise.init(
        db_url=DATABASE_URL,
        modules={'models': ['app.models']}
    )
    