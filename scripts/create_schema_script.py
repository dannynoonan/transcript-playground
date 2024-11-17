import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from tortoise import Tortoise, run_async
from app.database.connect import connect_to_database


# NOTE not verified since moving to /scripts dir
async def main():
    await connect_to_database()
    await Tortoise.generate_schemas()


if __name__ == '__main__':
    run_async(main())
