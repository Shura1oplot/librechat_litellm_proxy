#!/bin/sh

. ./.venv/bin/activate

[ -f .env ] && set -a && . ./.env && set +a

litellm \
    --config config.yaml \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --keepalive_timeout 60
