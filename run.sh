#!/usr/bin/env bash

# shellcheck disable=SC1091
[ -f .env ] && set -a && source .env && set +a

litellm \
    --config config.yaml \
    --port 4000 \
    --keepalive_timeout 60
