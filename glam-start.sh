#!/bin/bash

export DB_NAME=$(cat /run/secrets/GLAM2_DB_NAME)
export DB_USERNAME=$(cat /run/secrets/GLAM2_DB_USERNAME)
export DB_PASSWORD=$(cat /run/secrets/GLAM2_DB_PASSWORD)
export DB_HOST=$(cat /run/secrets/GLAM2_DB_HOST)
export DB_PORT=$(cat /run/secrets/GLAM2_DB_PORT)
export AWS_ACCESS_KEY_ID=$(cat /run/secrets/GLAM2_AWS_ACCESS_KEY_ID)
export AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/GLAM2_AWS_SECRET_ACCESS_KEY)
export AWS_REGION=$(cat /run/secrets/GLAM2_AWS_REGION)
export AWS_DEFAULT_REGION=$AWS_REGION
export GTM_CONTAINER=$(cat /run/secrets/GLAM2_GTM_CONTAINER)


gunicorn --timeout 120 --workers 4 --worker-class gevent --bind 0.0.0.0:8050 app:server