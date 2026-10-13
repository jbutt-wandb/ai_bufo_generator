#!/usr/bin/env bash
# Mirror the local repo to the cluster project dir (~/bufo_gen).
# .env IS synced on purpose: the cluster app needs the same keys.
set -euo pipefail
cd "$(dirname "$0")/.."

rsync -avz --delete \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude generated_videos \
  --exclude public \
  --exclude uploads \
  --exclude reference \
  --exclude logs \
  --exclude wandb \
  ./ coreweave-login:bufo_gen/

# uv lives in ~/.local/bin on the cluster, which non-interactive shells don't add to PATH.
echo "Synced. Next: ssh coreweave-login 'cd bufo_gen && ~/.local/bin/uv sync'"
