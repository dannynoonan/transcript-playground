from tortoise import Tortoise, run_async
from database.connect import connect_to_database


async def main():
    await connect_to_database()
    await Tortoise.generate_schemas()


if __name__ == '__main__':
    run_async(main())
