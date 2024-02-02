from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    pg_user: str
    pg_password: str
    pg_host: str
    pg_port: int
    pg_db_name: str

    es_user: str
    es_password: str
    es_host: str
    es_port: int

    kb_password: str
    kb_port: int

    es_stack_version: str
    es_cluster_name: str
    es_license: str
    es_mem_limit: int

    openai_api_key: str
    model_config = SettingsConfigDict(env_file=".env")

# TODO use lru_cache with fastapi Depends
# per https://fastapi.tiangolo.com/advanced/settings/#__tabbed_2_1
settings = Settings()

# https://levelup.gitconnected.com/handle-registration-in-fastapi-and-tortoise-orm-2dafc9325b7a
DATABASE_URL = f"postgres://{settings.pg_user}:{settings.pg_password}@{settings.pg_host}:{settings.pg_port}/{settings.pg_db_name}"
# DATABASE_URL = f"postgres://{settings.psql_user}@{settings.psql_host}:{settings.psql_port}/{settings.psql_db_name}"

TORTOISE_ORM = {
    "connections": {"default": DATABASE_URL},
    "apps": {
        "models": {
            "models": ["app.models", "aerich.models"],
            "default_connection": "default",
        },
    },
}
