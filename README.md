# Project overview

`transcript-playground` defines a simple, standardized data model into which any dialogue-driven television series can be transformed, persisted, and indexed, then builds text analytics / AI / ML features against that normalized/indexed transcript corpus.

As of this writing, properly ingested/normalized show transcripts can:
* combine character- and location-based faceting with free text bag-of-words search against scene dialogue and descriptions
* combine ES-native BOW-search with embeddings from pretrained Word2Vec and OpenAI transformer models to expand search and recommendation features 
* leverage OpenAI embeddings for basic classification and clustering operations

# Tech stack overview
* `FastAPI`: lightweight async python API framework
* `Postgres` / `Toroise` / `Pydantic`: RDBMS and ORM framework 
* `ElasticSearch` / `elasticsearch-dsl`: lucene-based index and ORM-style utilities
* `BeautifulSoup`: text parser for transforming raw HTML to normalized objects 
* `Pandas` / `NumPy` / `Scikit-learn` / `NLTK`: data / text analytics / ML tool kits
* `Word2Vec` / `OpenAI`: pre-trained language / embedding models 

# Setup and run 

Basic workflow for getting the app up and running after cloning the repo locally.

## Set properties in .env
```
cp .env.example .env
```

Assign values to all vars in newly created `.env` file. These vars will be exposed within project modules as `settings` props defined in `app/config.py`. If you build and run with `docker` (as opposed to running locally via `uvicorn`) then these props will also be substituted into your `docker-compose.yml` file.

Defaults are suggested for several properties, but you will need to assign your own Elasticsearch and Postgres credentials. 

Also note that, when using the `docker-compose.yml` setup, `HOST` properties should be assigned to docker container names, rather than being set to `localhost`.

## Data source configuration

Elasticsearch and Postgres require either local setup or docker setup. Both approaches will generate user credentials that can be assigned to corresponding `.env` vars.

This README does not explain how to get Elasticsearch or Postgres running, but these quick overviews and links might be helpful. 

### Elasticsearch

[Installing Elasticsearch with Docker](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html) gives a fantastic overview of how to get Elasticsearch and Kibana up and running in docker, with proper keys and credentials. Those credentials can be used for running Elasticsearch from `docker-compose.yml` as well. 

### Postgres

This [FastAPI-RDBMS integration](https://fastapi.tiangolo.com/tutorial/sql-databases/) walk-thru covers Postgres as a general example of RDBMS, and this [Tortoise quickstart](https://tortoise.github.io/getting_started.html) and [FastAPI-TortoiseORM integration tutorial](https://medium.com/@talhakhalid101/python-tortoise-orm-integration-with-fastapi-c3751d248ce1) will get you up to speed on how Postgres and Tortoise ORM are set up and used in the project. Similar to the Elasticsearch setup, user/password credentials need to be added to `.env`.


## Option 1: Run app piece-meal via command-line docker and/or local install

If you've got Elasticsearch and Postgres running independently and configured in `.env`, then `transcript-playground` can be started locally using `venv` and `uvicorn`:
```
python -m venv venv
source venv/bin/activate
```

Install dependencies using `pip`:
```
pip install -r /path/to/requirements.txt
```

Run/restart the app using `uvicorn`:
```
uvicorn main:app --reload
```

(If you're content running within `venv` and `uvicorn` you can skip past the "Run app using docker compose" section to "Verify app is running.")


## Option 2: Run app using docker compose

### A note on docker versions

Docker versioning is confusing. A command-line version check on my Mac (OS 12.6.2) gets back this:
```
$ docker --version
Docker version 24.0.7, build afdd53b
```

I'm running Docker Desktop, and behind the "Settings" button and under "Software Updates" I see:
```
You're currently on version 4.26.1 (131620). The latest version is 4.27.1 (136059)
```

And all the `docker-compose.yml` examples I've come across specify `version: '3.8'` or thereabouts at the top.

Suffice to say: Results may vary depending on what versions and combinations of versions you are running.


### Building and starting

After setting user credentials and mapping data sources to docker containers in `.env`, try running:
```
docker compose up --build
```

With any luck this will:
* fetch dependencies, build, and start up the project's postgres `pg`, elasticsearch `es`, kibana `kib`, and webapp `web` services
* expose, authenticate, and configure all port interdependencies between services
* spin up data volumes that live on after a container is stopped / can be accessed again when container is restarted

More than likely you'll run into a snag, perhaps even with the `docker compose up` syntax instead of `docker-compose up` syntax (which relates to a recent-ish version change). I can't anticipate all the variations for reasons noted above, but I'd love to hear feedback on what works and doesn't work.


### Shutting down, restarting, rebuilding

Shutting down and restarting: 
```
docker compose down
docker compose up
```

Depending on the nature of your code changes, you may need to:
* tack `-v` to the end of `docker compose down -v` to remove volumes
* tack `--build` to the end on `docker compose up --build` to reflect dependency changes in `requirements.txt` or renaming/remapping of services in `docker-compose.yml` or properties set in `.env`


## Verify app is running

Verify app is up by hitting http://127.0.0.1:8000/ and seeing "Welcome to transcript playground" or hitting http://127.0.0.1:8000/docs#/ for the OpenAPI/Swagger listing of available endpoints.


### Verify basic ETL -> ES Writer workflow

Before launching into API endpoint detail below, verify end-to-end connectivity and functionality by hitting these endpoints in the following sequence. 

| Endpoint        | Action           | Response  |
| ------------- |-------------| -----|
| `/esw/init_es` | initializes `transcript` index mappings (verifiable in Kibana DevTools at http://0.0.0.0:5601/app/dev_tools#/console with `GET /transcripts/_mapping`) | "success" |
| `/etl/copy_episode_listing/TNG` | copies episode listing HTML for `show_key=TNG` to local `source/` dir | file paths, raw html for episode listing |
| `/etl/load_episode_listing/TNG?write_to_db=True` | loads copied transcript sources HTML for `show_key=TNG` from `source/` into Postgres (if `write_to_db=True` flag is left off, code will run without writing to db) | episode listing counts and contents |
| `/etl/copy_transcript_sources/TNG` | copies transcript sources HTML for `show_key=TNG` to local `source/` dir | file paths, raw html for transcript sources |
| `/etl/load_transcript_sources/TNG?write_to_db=True` | loads copied transcript sources HTML for `show_key=TNG` from `source/` into Postgres (if `write_to_db=True` flag is left off, code will run without writing to db) | transcript sources counts and contents |
| `/etl/copy_transcript_from_source/TNG/150` | copies episode transcript HTML for `show_key=TNG` and `episode_key=150` to local `source/` dir | file paths, raw html for episode |
| `/etl/load_transcript/TNG/150?write_to_db=True` | loads copied episode transcript HTML for `show_key=TNG` and `episode_key=150` from `source/` into Postgres (if `write_to_db=True` flag is left off, code will run without writing to db) | episode transcript data normalized for db write |
| `/esw/index_episode/TNG/150` | fetches episode transcript data for `show_key=TNG` and `episode_key=150` from Postgres and writes it to `transcripts` index | episode transcript data normalized for es write |
| `/web/episode/TNG/150` | leverages various `/esr` endpoints to fetche episode transcript data for `show_key=TNG` and `episode_key=150` from `transcripts` index and render to web page via `jinja` HTML template | web page displaying episode data |

All endpoints follow `http://127.0.0.1:8000` (e.g. http://127.0.0.1:8000/etl/copy_episode_listing/TNG) assuming you're running the app on port 8000.

### Tests

Tests currently cover a subset of core data transformations in the ETL pipeline. More to be done here.
```
pytest -v tests.py
```

# App details 

## Metadata overview

### Show metadata

* `show_key`: unique identifier that is shorthand for a distinct show, even if that show is part of a broader franchies (e.g. "Star Trek: The Original Series" has `show_key='TOS'`, while "Star Trek: The Next Generation" has `show_key='TNG'`)
* `external_key` or `episode_key`: unique identifier for a specific episode of a given `show_key`, ideally derived from an agreed-upon external data source or else generated from episode titles

### NLP metadata

A mapping of predefined `model_vendor` and `model_version` values and associated metadata are managed in the `WORD2VEC_VENDOR_VERSIONS` and `TRANSFORMER_VENDOR_VERSIONS` variables in `nlp/nlp_metadata.py`.

**Note:** Word2Vec models are massive and are not stored in the project github repo. If you decide to leverage any of the Word2Vec embedding functionality, the `nlp/nlp_metadata.py` will guide you to external resources where you can download pretrained Word2Vec model files. Once downloaded, rename a given model file to match the `versions` keys in the `WORD2VEC_VENDOR_VERSIONS` variable (again using `nlp/nlp_metadata.py` as a guide), then store the renamed model file(s) to one of the following directories:
* `w2v_models/fasttext`
* `w2v_models/glove`
* `w2v_models/webvectors`

You only need to download and configure Word2Vec language models that you intend to use, and you can ignore those you don't intend to use. Getting up and running with `model_vendor:'glove'` / `model_version:'6B300d'` does not require you to download or rename any other `glove` model versions, or models from any other vendor.

If you decide to experiment with the `/esw/build_embeddings_model` endpoint (described below) you will also need to create:
* `w2v_models/homegrown`

OpenAI has nice clean APIs for generating Transformer embeddings, so rather than downloading language models locally you will need to add your own `OPENAI_API_KEY` param value to `.env`.


## ETL source/ directories

Before episode listing and transcript data are loaded into `transcript_db` (Postgres), it is first copied from external html files and staged in local `source/` subdirectories. This protects your db data from being paved over if you refresh/reload episode data following an unexpected change in an external content source (reformatting of a Wikipedia page, a fan site changing its url structure, etc).

Staging files and directories are not tracked in `git`, but as you tick through the `/etl/copy_X` endpoints (outlined below) you will begin to see raw html files populating local `source/` subdirectories. Re-running the same `/etl/copy_X` endpoint more than once will bump any file into an adjacent `backup/` directory before overwriting it.

Note that re-running the same `/etl/copy_X` endpoint twice will effectively overwrite the overwrite, so this "fall back to previous version" scheme is primitive. At some point I hope to highlight any differences as backups are made, maybe a simple report illustrating the diff between `source/` files and their corresponding `backup/` files.

(There is also a `/backup_db` endpoint with a similar aim of protecting local data from being wiped out by upstream data source changes or outages.)


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


### 'ETL' endpoints: source and load data into Postgres  

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
        

### 'ES Writer' endpoints: populate ElasticSearch 

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


### 'ES Reader' endpoints: query ElasticSearch  

ES Reader `/esr` endpoints provide most of the core functionality of the project, since ElasticSearch houses free text, facet-oriented, and vectorized representations of transcript data that span the gamut of search, recommendation, classification, clustering, and other AI/ML-oriented features. I won't continually track the feature set in the README, but at the time of this writing the endpoints generally broke down into these buckets:
* show listing and metadata lookups (key- or id-based fetches)
* free text search (primarily of dialogue, but also of title and description fields)
* faceted search in combination with free text search (keying off of selected characters, locations, or seasons)  
* aggregations (effectively your "count(*) / group by" type queries keying off facets like characters, locations, or seasons)
* vector search (K-Nearest Neighbor cosine similarity comparison of search queries and documents as represented in multi-dimensional vector space)
* AI/ML functionality like MoreLikeThis and termvectors (native to ElasticSearch) or clustering and classification (leveraging embeddings data as inputs to ML model training)


### 'Web' endpoints: render web pages 

Webpage-rendering endpoints are 'front-end' consumers of the other 'back-end' endpoints, specifically of the 'ES Reader' endpoints. These 'Web' endpoints generate combinations of `/esr` requests, package up the results, and feed them into HTML templates that offer some bare-bones UI functionality.


## Analytics

I've set up a rudimentary query input -> response ranking pipeline to evaluate the performance of various Word2Vec and Transformer language models. The 'test data' are short summaries of episodes pulled from various data sources (fan sites, reviews portals, etc), and 'success' is measured by how well an episode description does at matching the episode being described in search results.

The current ad hoc setup is built around `show_key:'TNG'`, but the overarching workflow is generic and--with some whittling--could be expanded to cover other shows.

Analytics processes are triggered as scripts rather than via API endpoints:
* `load_description_sources.py`: fetches episode descriptions from external data sources, maps them to episode keys, and writes them to a freeze-dried pandas dataframe stored as csv
* `generate_vsearch_rankings.py`: invokes the `/esr/vector_search` endpoint for each externally-sourced episode description, determines how well the described episode ranks in search results, and writes that ranking/performance output to a freeze-dried pandas dataframe stored as csv


## Db migrations

`transcript-playground` is configured for Postgres migrations using Toroise ORM and Aerich. Migrations are executed at app startup. The following steps describe the process for re-initializing migrations and for executing them going forward.

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


# Next phases

## Lateral growth: adding new shows / ingestion transformers
Ideally, the transcript ingest and normalization code should be easily extensible, allowing for new ETL parsers to be easily added for new shows. Some restructuring of the ETL code is needed for the onboarding of new shows / new transcript parsers to be smoother / less ad hoc.

## Vertical growth: expanding text analytics feature set
On deck: AI/ML models trained using embedding data generated via OpenAI. Interactive data visualization by integrating Plotly/Dash into FastAPI/Flask.


# Ongoing development

