#!/bin/bash
set -e
cd "$(dirname "$0")"

BRANCH=$(git branch --show-current)
echo "Watching branch: $BRANCH"
echo "Auto-pull every 30s + Flask hot-reload"
echo ""

(
  while true; do
    git fetch --quiet origin "$BRANCH" 2>/dev/null
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo "$LOCAL")
    if [ "$LOCAL" != "$REMOTE" ]; then
      echo ""
      echo ">>> Mudanças detectadas, fazendo pull..."
      git pull --quiet origin "$BRANCH"
      echo ">>> Atualizado para $(git rev-parse --short HEAD)"
    fi
    sleep 30
  done
) &

PULL_PID=$!
trap "kill $PULL_PID 2>/dev/null; exit" INT TERM EXIT

python3 web/app.py
