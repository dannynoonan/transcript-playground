# Project overview

This project defines a simple, standardized data model into which any dialogue-driven television series can be transformed, persisted, and indexed, then builds text analytics / AI / ML features against that normalized/indexed transcript data.

As of this writing, properly ingested/normalized show transcripts can leverage character- and location-based faceting, aggregation, and free text search against the show's transcript corpus. Search and recommendation features combine bag-of-words search native to ElasticSearch with embeddings from pretrained Word2Vec and OpenAI transformer models. OpenAI embeddings are also leveraged for basic classification and clustering operations.

## Next phases

### Lateral growth: adding new shows / ingestion transformers
Ideally, the transcript ingest and normalization code should be easily extensible, allowing for new ETL parsers to be easily added for new shows. Some restructuring of the ETL code is needed for the onboarding of new shows / new transcript parsers to be smoother / less ad hoc.

### Vertical growth: expanding text analytics feature set
On deck: AI/ML models trained using embedding data generated via OpenAI. Interactive data visualization by integrating Plotly/Dash into FastAPI/Flask.


# Tech stack overview
* `FastAPI`: lightweight async python API framework
* `Postgres`/`Toroise`/`Pydantic`: RDBMS and ORM framework 
* `ElasticSearch`/`elasticsearch-dsl`: lucene-based index and ORM-style utilities
* `BeautifulSoup`: text parser for transforming raw HTML to normalized objects 
* `Pandas`/`NumPy`/`Scikit-learn`/`NLTK`: data / text analytics / ML tool kits
* `Word2Vec`/`OpenAI`: pre-trained language / embedding models 


# Setup

## Install dependencies
```
pip install -r /path/to/requirements.txt
```

## Run elasticsearch

`#TODO` set up `Dockerfile` ([Install Elasticsearch with Docker](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html) has carried me so far)

## Initialize Postgres / Toroise ORM / Aerich migrations

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

## Config

Copy the contents of `.env.example` to a root-level `.env` file, then set valid values for each parameter based on your own local configuration.

## Tests
```
pytest -v tests.py
```

## Run app 
```
uvicorn main:app --reload
```

Verify app is up by hitting http://127.0.0.1:8000/docs#/.


# Metadata overview

## Show metadata

* `show_key`: unique identifier that is shorthand for a distinct show, even if that show is part of a broader franchies (e.g. "Star Trek: The Original Series" has `show_key='TOS'`, while "Star Trek: The Next Generation" has `show_key='TNG'`)
* `external_key` or `episode_key`: unique identifier for a specific episode of a given `show_key`, ideally derived from an agreed-upon external data source or else generated from episode titles

## NLP metadata

A mapping of predefined `model_vendor` and `model_version` values and associated metadata are managed in the `WORD2VEC_VENDOR_VERSIONS` and `TRANSFORMER_VENDOR_VERSIONS` variables in `nlp/nlp_metadata.py`.

**Note:** Word2Vec models are massive and are not stored in the project github repo. If you decide to leverage any of the Word2Vec embedding functionality, the `nlp/nlp_metadata.py` will guide you to external resources where you can download pretrained Word2Vec model files. Once downloaded, rename a given model file to match the `versions` keys in the `WORD2VEC_VENDOR_VERSIONS` variable (again using `nlp/nlp_metadata.py` as a guide), then store the renamed model file(s) to one of the following directories:
* `w2v_models/fasttext`
* `w2v_models/glove`
* `w2v_models/webvectors`

You only need to download and configure Word2Vec language models that you intend to use, and you can ignore those you don't intend to use. Getting up and running with `model_vendor:'glove'` / `model_version:'6B300d'` does not require you to download or rename any other `glove` model versions, or models from any other vendor.

If you decide to experiment with the `/esw/build_embeddings_model` endpoint (described below) you will also need to create:
* `w2v_models/homegrown`

OpenAI has nice clean APIs for generating Transformer embeddings, so rather than downloading language models locally you will need to add your own `OPENAI_API_KEY` param value to `.env`.


# API overview

* OpenAPI: http://127.0.0.1:8000/docs#/
* Redoc: http://127.0.0.1:8000/redoc
* Raw json: http://127.0.0.1:8000/openapi.json

API endpoints currently do most of the work in the app. Endpoints live within 4 router zones:
* External sourcing / db writing (etl): Sourcing html from the web, saving it as raw text, transforming it to tortoise/pydantic model objects, and writing objects to Postgres
* ElasticSearch writing (esw): Fetching objects from Postgres or documents from ElasticSearch, transforming and writing documents to ElasticSearch
* ElasticSearch reading (esr): Searching documents and aggregating counts or statistics from ElasticSearch
* WebApp (web): Accept user input via jinja html templates and translate these into requests for "backend" endpoints (typically esr)

A few endpoints that don't fall cleanly into those four buckets still live in `main.py` for the time being.


## 'ETL' endpoints" to source and load data into Postgres  

ETL endpoints fall into two buckets: `/copy_X` and `/load_X`
* `/etl/copy_X` endpoints: fetch html content from external sources (defined in `show_metadata.py`) and store as raw text files to local `source/` directory.
    * Since external data sources are beyond our control, this step is a safeguard against overwriting previously-loaded Postgres/ElasticSearch data, effectively functioning as a staging data layer. 
    * Contents of `source/` are not tracked in github, so the following directories need to be created: 
        * `source/episodes/`
        * `source/episode_listings/`
        * `source/transcript_sources/`
    * `/etl/copy_X` endpoints can be run in any order, but logically they proceed in this sequence:
        * `/etl/copy_episode_listing/{show_key}`: copies html of external episode listing page to `source/episode_listings/` 
        * `/etl/copy_transcript_sources/{show_key}`: copies html of external transcript url listing page to `source/transcript_sources/`
        * `/etl/copy_transcript_from_source/{show_key}/{episode_key}` copies html of external episode page to `source/episodes/` (it makes sense to spot-check that this works on a couple of episodes before running the bulk `/etl/copy_all_transcripts_from_source/` operation)
        * `/etl/copy_all_transcripts_from_source/{show_key}`: bulk run of `/etl/copy_transcript_from_source` for all episodes of a given show
* `/etl/load_X` endpoints: extract raw html written to `source/` directory during `/etl/copy_X` actions, transform these into Tortoise-ORM mapped objects, and write them to Postgres `transcript_db`.
    * `/etl/load_X` endpoints should be run in this order:
        * `/etl/load_episode_listing/{show_key}`: initializes `Episode` db entities 
        * `/etl/load_transcript_sources/{show_key}`: initializes `TranscriptSource` db entities, has dependencies on `Episode` db entities
        * `/etl/load_transcript/{show_key}/{episode_key}`: transform and load `Scene` and `SceneEvent` db entites as children of `Episode` db entities, has dependencies on `Episode` db entities and `TranscriptSource` db entities (it makes sense to spot-check that this works on a couple of episodes before running the bulk `/etl/load_all_transcripts` operation)
        * `/etl/load_all_transcripts/{show_key}`: bulk run of `/etl/load_transcript` for all episodes of a given show
        

## 'ES Writer' endpoints: to populate ElasticSearch 

**Important:** First run the `/esw/init_es` endpoint to generate mappings for the "transcripts" index. If you do not do this, index writes may proceed without issuing warnings, but certain index read operations will fail where data has been mapped to default field types.

Primary endpoints for es index writing:
* `/esw/index_episode/{show_key}/{episode_key}`: fetch episode transcript from Postgres, transform Tortoise object to ElasticSearch object, and write it to ElasticSearch index
* `/esw/index_all_episodes/{show_key}`: bulk run of `/esw/index_episode` for all episodes of a given show 

Auxiliary endpoints for es index writing:
* `/esw/populate_focal_speakers/{show_key}`: for each episode, aggregate the number of lines spoken per character, then store the top 3 characters in their own index field 
* `/esw/populate_focal_locations/{show_key}`: for each episode, aggregate the number of scenes per location, then store the top 3 locations in their own index field 

Endpoints for writing vector embeddings to es index:
* `/esw/build_embeddings_model/{show_key}` (not in use / experimental): goes thru the motions of building a language model using Word2Vec, but limits training data to a single show's text corpus, resulting in a uselessly tiny model
* `/esw/populate_embeddings/{show_key}/{episode_key}/{model_vendor}/{model_version}`: generate vector embedding for episode using pre-trained, publicly available Word2Vec and Transformer models
* `/esw/populate_all_embeddings/{show_key}/{model_vendor}/{model_version}`: bulk run of `/esw/populate_embeddings` for all episodes of a given show


## 'ES Reader' endpoints: to query ElasticSearch  

ES Reader `/esw` endpoints provide most of the core functionality of the project, since ElasticSearch houses free text, facet-oriented, and vectorized representations of transcript data that span the gamut of search, recommendation, classification, clustering, and other AI/ML-oriented features. I won't continually track the feature set in the README, but at the time of this writing the endpoints generally broke down into these buckets:
* show listing and metadata lookups (key- or id-based fetches)
* free text search (primarily of dialogue, but also of title and description fields)
* faceted search in combination with free text search (keying off of selected characters, locations, or seasons)  
* aggregations (effectively your "count(*) / group by" type queries keying off facets like characters, locations, or seasons)
* vector search (K-Nearest Neighbor cosine similarity comparison of search queries and documents as represented in multi-dimensional vector space)
* AI/ML functionality like MoreLikeThis and termvectors (native to ElasticSearch) or clustering and classification (leveraging embeddings data as inputs to ML model training)


## 'Web' endpoints: render web pages 

Webpage-rendering endpoints are 'front-end' consumers of the other 'back-end' endpoints, specifically of the 'ES Reader' endpoints. These 'Web' endpoints generate combinations of `/esr` requests, package up the results, and feed them into HTML templates that offer some bare-bones UI functionality.
