#!/bin/bash

NOW=$(date +"%Y-%m-%dT%T.%3N")
PGDUMP="/Applications/Postgres.app/Contents/Versions/latest/bin/pg_dump -c"
DB=transcript_db
BAKFILE="database/backups/$DB-$NOW.sql"
$PGDUMP $DB > $BAKFILE