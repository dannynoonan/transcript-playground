from tortoise import Tortoise

# DATABASE_URL="postgres://andyshirey@localhost:5432/transcript_db?sslmode=disable"
DATABASE_URL="postgres://andyshirey@localhost:5432/transcript_db"

async def connect_to_database():
    await Tortoise.init(
        db_url=DATABASE_URL,
        modules={'models': ['app.models']}
    )