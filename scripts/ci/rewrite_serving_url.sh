#!/usr/bin/env bash
# Rewrite the flaime-serving SSH URL to HTTPS+token for CI environments.
#
# uv uses libgit2, which ignores git config insteadOf rules, so the URL must be
# rewritten directly in pyproject.toml and uv.lock. Mirrors FLAIME's
# scripts/ci/rewrite_phonet_url.sh (DEVOPS-06) for the same reason.
#
# Usage: FLAIME_SERVING_PAT=<token> bash scripts/ci/rewrite_serving_url.sh

set -euo pipefail

if [ -z "${FLAIME_SERVING_PAT:-}" ]; then
    echo "Error: FLAIME_SERVING_PAT environment variable is not set" >&2
    exit 1
fi

sed -i "s|ssh://git@github.com/IndigiGenius/flaime-serving.git|https://x-access-token:${FLAIME_SERVING_PAT}@github.com/IndigiGenius/flaime-serving.git|g" pyproject.toml
[ -f uv.lock ] && sed -i "s|ssh://git@github.com/IndigiGenius/flaime-serving.git|https://x-access-token:${FLAIME_SERVING_PAT}@github.com/IndigiGenius/flaime-serving.git|g" uv.lock
echo "flaime-serving URL rewritten to HTTPS+token"
