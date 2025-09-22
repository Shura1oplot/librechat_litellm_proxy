#!/bin/sh

# shellcheck source=/dev/null
. ./.venv/bin/activate

# shellcheck source=/dev/null
[ -f .env ] && set -a && . ./.env && set +a

litellm \
    --config config.yaml \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --keepalive_timeout 60
