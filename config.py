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


settings = Settings()


# TODO remove this and use lru_cache with fastapi Depends
# per https://fastapi.tiangolo.com/advanced/settings/#__tabbed_2_1
