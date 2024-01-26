# transcript-playground

This project defines a simple, standardized data model into which any dialogue-driven television series can be transformed, persisted, and indexed, then builds features against that normalized/indexed transcript data.

The idea is that, for any show having a reasonably well-formatted public transcript source, it shouldn't be much trouble to write ingest parsers/transformers mapping that public transcript data and episode listing metadata to the `transcript-playground` data model, and that any properly-loaded show can leverage the same text analytics feature suite.

Currently, although the standardized data model feels relatively solid, the design of the transcript ingest code requires some restructuring to make it more extensible and less ad hoc.

The feature set awaiting any properly-ingested transcript content is very much under development, but as of this writing incorporates character- and location-based faceting, aggregation, and free text dialogue search using standard bag-of-words indexing, embedding-driven search using trained Word2Vec and OpenAI transformer models, and basic classification and clustering operations using OpenAI embeddings (but it feels like the sky is the limit now that the basic building blocks are wired up). 


## Tech stack overview
* FastAPI: lightweight async python API framework
* Postgres/Toroise/Pydantic: RDBMS and ORM framework 
* ElasticSearch/elasticsearch-dsl: lucene-based index and ORM-style utilities
* BeautifulSoup: text parser for transforming raw HTML to normalized objects 
* Pandas/NumPy/Scikit-learn/NLTK: data / text analytics / ML tool kits


## Setup

### Run elasticsearch

Huh, just realized I don't have a `Dockerfile` defined yet, but [Install Elasticsearch with Docker](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html) has carried me so far. Big `#TODO` here.

### Initialize Postgres / Toroise ORM / Aerich migrations

* Create `migrations/` dir and `pyproject.toml` file from settings in TORTOISE_ORM dict:
    ```
    aerich init -t config.TORTOISE_ORM
    ```

* Create app migration destination `migrations/models/` and generate baseline schema from `models.py`:
    ```
    aerich init-db
    ```

* Generate schema migration file from altered `models.py`:
    ```
    aerich migrate --name <migration_name>
    ```

* Execute migration file:
    ```
    aerich upgrade
    ```

Reference: https://tortoise.github.io/migration.html

### Tests
```
pytest -v tests.py
```

### Run app 
```
uvicorn main:app --reload
```


## Metadata overview

* `show_key`: unique identifier that is shorthand for a distinct show, even if that show is part of a broader franchies (e.g. "Star Trek: The Original Series" has `show_key='TOS'`, while "Star Trek: The Next Generation" has `show_key='TNG'`)
* `external_key` or `episode_key`: unique identifier for a specific episode of a given `show_key`, ideally derived from an agreed-upon external data source or else generated from episode titles


## API overview

* OpenAPI: http://127.0.0.1:8000/docs#/
* Redoc: http://127.0.0.1:8000/redoc
* Raw json: http://127.0.0.1:8000/openapi.json

API endpoints currently do most of the work in the app. Endpoints live within 4 router zones:
* External sourcing / db writing (etl): Sourcing html from the web, saving it as raw text, transforming it to tortoise/pydantic model objects, and writing objects to Postgres
* ElasticSearch writing (esw): Fetching objects from Postgres or documents from ElasticSearch, transforming and writing documents to ElasticSearch
* ElasticSearch reading (esr): Searching documents and aggregating counts or statistics from ElasticSearch
* WebApp (web): Accept user input via jinja html templates and translate these into requests for "backend" endpoints (typically esr)

A few endpoints that don't fall cleanly into those four buckets still live in `main.py` for the time being.


### Load data into Postgres using ETL endpoints

ETL endpoints fall into two buckets: `/copy_X` and `/load_X`
* `/etl/copy_X` endpoints: fetch html content from external sources (defined in `show_metadata.py`) and store as raw text files to local `source/` directory.
    * Since external data sources are beyond our control, this step is a safeguard against overwriting previously-loaded Postgres/ElasticSearch data, effectively functioning as a staging data layer. 
    * Contents of `source/` are not tracked in github, so the following directories need to be created: 
        * `source/episodes/`
        * `source/episode_listings/`
        * `source/transcript_sources/`
    * `/etl/copy_X` endpoints can be run in any order, but logically they proceed in this sequence:
        * `/etl/copy_episode_listing/{show_key}`
        * `/etl/copy_transcript_sources/{show_key}`
        * Individual episodes can be copied using `/etl/copy_transcript_from_source/{show_key}/{episode_key}` (it makes sense to spot-check that this works on a couple of episodes before running the bulk `copy_all_transcripts` operation)
        * `/etl/copy_all_transcripts_from_source/{show_key}`
* `/etl/load_X` endpoints: extract raw html written to `source/` directory during `/etl/copy_X` actions, transform these into Tortoise-ORM mapped objects, and write them to Postgres `transcript_db`.
    * `/etl/load_X` endpoints should be run in this order:
        * `/etl/load_episode_listing/{show_key}`: initializes `Episode` db entities 
        * `/etl/load_transcript_sources/{show_key}`: initializes `TranscriptSource` db entities, has dependencies on `Episode` db entities
        * `/etl/load_transcript/{show_key}/{episode_key}`: transform and load `Scene` and `SceneEvent` db entites as children of `Episode` db entities (it makes sense to spot-check that this works on a couple of episodes before running the bulk `load_all_transcripts` operation)
        * `/etl/load_all_transcripts/{show_key}`: transforms and loads all transcripts for a given show
        

### Populate ElasticSearch index using ES Writer endpoints

*Important:* First run the `/esw/init_es` endpoint to generate mappings for the "transcripts" index. If you do not do this, data ingest may proceed without issuing warnings, but granular query operations will fail due to invalid default field types being used instead of declared field types.

Primary es index writing endpoints:
* `/esw/index_episode/{show_key}/{episode_key}`
* `/esw/index_all_episodes/{show_key}` 

Auxiliary es index writing endpoints:
* `/esw/populate_focal_speakers/{show_key}`
* `/esw/populate_focal_locations/{show_key}`

Embeddings data writing endpoints:
* `/esw/build_embeddings_model/{show_key}`
* `/esw/populate_embeddings/{show_key}/{episode_key}/{model_version}/{model_vendor}`
* `/esw/populate_all_embeddings/{show_key}/{model_version}/{model_vendor}`


### Query ElasticSearch index using ES Reader endpoints





### Render web pages using Web endpoints
