# Project overview

`transcript-playground` defines a simple, standardized data model into which any dialogue-driven television series can be transformed, persisted, and indexed, then builds text analytics / AI / ML features against that normalized/indexed transcript corpus.

As of this writing, properly ingested/normalized series transcripts can:
* leverage Elasticsearch-native bag-of-words indexing and OpenAI (or Word2Vec) embeddings to enable transcript search, faceting, and item-based recommendation features 
* leverage Bertopic and OpenAI embeddings for basic classification and clustering operations
* render as interactive Plotly figures served via Dash web pages

# Tech stack overview
* `FastAPI`: lightweight async python API framework
* `Plotly` / `Dash`: data visualization and associated web delivery frameworks
* `ElasticSearch` / `elasticsearch-dsl`: lucene-based index and ORM-style utilities
* `Postgres` / `Toroise` / `Pydantic`: RDBMS and ORM framework 
* `Pandas` / `NumPy` / `Scikit-learn` / `NLTK`: data / text analytics / ML tool kits
* `OpenAI` / `Word2Vec`: pre-trained language / embedding models 
* `Bertopic`: clustering features built around BERT language model
* `Airflow`: orchestration manager for data ingestion, mapping, and indexing
* `BeautifulSoup`: text parser for transforming raw HTML to normalized objects 
* `Docker`: service containerization / dependency / deployment manager

# Setup and run 

Basic workflow for getting the app up and running after cloning the repo locally.

## Set properties in .env
```
cp .env.example .env
```

Assign values to all vars in newly created `.env` file. These vars will be exposed within project modules as `settings` props defined in `app/config.py`. If you build and run with `docker` (as opposed to running locally via `uvicorn`) then these props will also be substituted into `docker-compose` files.

Defaults are suggested for several properties, but you will need to assign your own Elasticsearch, Postgres, OpenAI, and Airflow credentials/keys. 

Also note that, when using the `docker-compose.yml` setup, `HOST` properties should be assigned to docker container names, rather than being set to `localhost`.

## Data source configuration

Elasticsearch and Postgres require either local setup or docker setup. Both approaches will generate user credentials that can be assigned to corresponding `.env` vars.

This README does not explain how to get Elasticsearch or Postgres running, but these quick overviews and links might be helpful. 

### Elasticsearch

[Installing Elasticsearch with Docker](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html) gives a fantastic overview of how to get Elasticsearch and Kibana up and running in docker, with proper keys and credentials. Those credentials can be used for running Elasticsearch locally or via `docker-compose.yml` (or other `docker-compose-*` variants). 

### Postgres

This [FastAPI-RDBMS integration](https://fastapi.tiangolo.com/tutorial/sql-databases/) walk-thru covers Postgres as a general example of RDBMS, and this [Tortoise quickstart](https://tortoise.github.io/getting_started.html) and [FastAPI-TortoiseORM integration tutorial](https://medium.com/@talhakhalid101/python-tortoise-orm-integration-with-fastapi-c3751d248ce1) will get you up to speed on how Postgres and Tortoise ORM are set up and used in the project. Similar to the Elasticsearch setup, user/password credentials need to be added to `.env`.


## Option 1: Run all services using docker compose

### Building and starting 

* Set up Elasticsearch and Postgres credentials and container names/mappings in `.env`
* Run `docker compose up --build`

With any luck this will:
* fetch dependencies, build, and start up the project's postgres `pg`, elasticsearch `es`, kibana `kib`, webapp `web`, redis `redis`, and various Airflow services
* expose, authenticate, and configure all port interdependencies between services
* spin up data volumes that live on after a container is stopped / can be accessed again when container is restarted

Once built, minor changes should only require `docker compose up` and `docker compose down` between restarts.

*Recommended:* To start up all services except Airflow, run `docker compose -f docker-compose-web.yml up --build`

### Troubleshooting

As you are populating your `.env` properties and configuring your initial setup, you may need to tear down and rebuild a few times to get it right. A couple pointers:
* tack `--build` at the end of `docker compose up --build` to reload `requirements.txt` dependencies or rename/remap services in `docker-compose.yml` or properties in `.env`
* tack `-v` to the end of `docker compose down -v` to remove volumes (can be necessary for authentication/certs regeneration as well as database / index rebuilds)

Fully rebuilding and wiping out volumes can be costly in terms of processing time and lost data, but during initial setup it can be helpful to wipe the slate clean. Certs in particular have a way of hanging around unless forcibly evicted and regenerated after an auth setup change.


## Option 2: Run app piece-meal via combination of moduler docker-compose file, command-line docker, and/or local install

There are several ways to get Elasticsearch and Postgres running:
* via `docker-compose-elk.yml`
* via command-line docker
* via local installation

When you've got Elasticsearch and Postgres running and configured in `.env`, then `transcript-playground` can be started locally using `venv` and `uvicorn`:
```
python -m venv myenv
source myenv/bin/activate
```

Install dependencies using `pip`:
```
pip install -r requirements.txt
```

Run/restart the app using `uvicorn`:
```
uvicorn app.main:app --reload
```


## Verify app is running

Verify app is up by hitting http://127.0.0.1:8000/ and seeing "Welcome to transcript playground" or hitting http://127.0.0.1:8000/docs#/ for the OpenAPI/Swagger listing of available endpoints.

If running the full suite of docker services, verify Airflow is up by hitting http://127.0.0.1:8080/ and clicking "DAGs" for a list of active data pipelines.


### Basic ETL workflow

Before launching into API endpoint detail below, verify end-to-end connectivity and functionality by hitting these endpoints in the following sequence. 

| FastAPI endpoint(s) | Airflow DAG | Description |
| ------------- |-------------| -----|
| `/esw/init_es` | N/A | N/A | Initialize es index mappings | 
| `/etl/copy_episode_listing/`, `/etl/copy_transcript_sources/TNG`, `/etl/copy_[all_]transcript[s]_from_source` | `copy_sources` | Copy episode listing metadata, transcript source metadata, and transcript text to local txt files |
| `/etl/load_episode_listing/`, `/etl/load_transcript_sources`, `/etl/load_[all_]transcript[s]` | `load_episodes` | Load episode listing metadata, transcript source metadata, and episode transcript data into transcript_db |
| `/esw/index_[all_]episode[s]`, `/esw/populate_focal_speakers`, `/esw/populate_focal_locations` | `index_episodes` | Fetch episodes from transcript_db and write to transcripts es index |
| `/esw/populate_[all_]episode_embeddings` | `populate_episode_embeddings` | Generate episode vector embeddings via pre-trained language model (e.g. OpenAI) and write to es |
| `/esw/populate_[all_]episode_relations` | `populate_episode_relations` | Generate item-based episode similarity via vector embeddings / KNN query, write listings to es |
| `/esw/index_topic_grouping`, `/esw/populate_topic_[grouping_]embeddings` | `index_topics` | Load topic data from csv into es, generate embeddings per topic description, write to es |
| `/esw/index_[all_]speaker[s]`, `/esw/populate_[all_]speaker_embeddings` | `index_speakers` | Load speaker metadata from files and write to es |
| `/esw/populate_[all_]episode_topics`, `/esw/populate_episode_topic_tfidf_scores` | `populate_episode_topics` | Map episodes to topics via vector search, using both 'absolute' and 'frequency-based' scoring, write mappings to es |
| `/esw/populate_[all_]speaker_topics` | `populate_speaker_topics` | Map speakers to topics (at episode-, season-, and series-level) via vector search, write mappings to es |
| `/esw/populate_bertopic_model_clusters` | `populate_narratives` | Split episodes up into speaker-interaction sub-narratives, generate Bertopic clusters using sub-narratives as document corpus, write to es and csv |


### Tests

Tests currently cover a subset of core data transformations in the ETL pipeline. More to be done here.
```
pytest -v tests.py
```

# App details 

## Metadata overview

### Show metadata

* `show_key`: unique identifier that is shorthand for a distinct show, even if that show is part of a broader franchise (e.g. "Star Trek: The Original Series" has `show_key='TOS'`, while "Star Trek: The Next Generation" has `show_key='TNG'`)
* `external_key` or `episode_key`: unique identifier for a specific episode of a given `show_key`, ideally derived from an agreed-upon external data source or else generated from episode titles

### NLP metadata

A mapping of predefined `model_vendor` and `model_version` values and associated metadata are managed in `TRANSFORMER_VENDOR_VERSIONS` and `WORD2VEC_VENDOR_VERSIONS` in `nlp_metadata.py`.

**OpenAI** has clean embeddings APIs that do not require complex downloading and installation. Simply generate your own OpenAI API key and set the `OPENAI_API_KEY` param in `.env`.

**Word2Vec models** (legacy) are massive and are not stored in the project github repo. If you decide to leverage any of the Word2Vec embedding functionality, `nlp_metadata.py` will guide you to external resources where you can download pretrained Word2Vec model files. Once downloaded, rename a given model file to match the `versions` keys in the `WORD2VEC_VENDOR_VERSIONS` variable (again using `nlp/nlp_metadata.py` as a guide), then store the renamed model file(s) to one of the following directories:
* `w2v_models/fasttext`
* `w2v_models/glove`
* `w2v_models/webvectors`


## Plotly / Dash

TODO

## Publishing 

* Animations: the `Dash` web framework does not support animation rendering, so animations must be generated and served independently as fully-baked html pages. To generate an animation html page, run the `/scripts/publish_animation.py` script or `publish_animation` DAG. The generated animation html file will be saved to `/static` for live site delivery.

* Wordclouds: Generate a wordcloud against an episode, season, or series by running the `/scripts/publish_wordcloud.py` script or `publish_wordcloud` DAG. The generated wordcloud image will be saved to `/static` for live site delivery.


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
