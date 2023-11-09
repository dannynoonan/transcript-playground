from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    es_user: str
    es_pass: str
    es_host: str
    es_port: int
    psql_user: str
    psql_host: str
    psql_port: int
    psql_db_name: str
    model_config = SettingsConfigDict(env_file=".env")

# TODO use lru_cache with fastapi Depends
# per https://fastapi.tiangolo.com/advanced/settings/#__tabbed_2_1
settings = Settings()

# https://levelup.gitconnected.com/handle-registration-in-fastapi-and-tortoise-orm-2dafc9325b7a
DATABASE_URL = f"postgres://{settings.psql_user}@{settings.psql_host}:{settings.psql_port}/{settings.psql_db_name}"

# TORTOISE_ORM = {
#     "connections": {"default": DATABASE_URL},
#     "apps": {
#         "models": {
#             "models": ["app.models"],
#             "default_connection": "default",
#         },
#     },
# }
