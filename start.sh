#!/usr/bin/env bash
# Avvia MIC RES Sicilia con gunicorn
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$APP_DIR/.venv/bin/activate"
export FLASK_APP=run.py
export FLASK_ENV=production
export PYTHONPATH="$APP_DIR/src:$PYTHONPATH"
cd "$APP_DIR"
exec "$APP_DIR/.venv/bin/gunicorn" \
    --bind 0.0.0.0:5555 \
    --workers 1 \
    --threads 8 \
    --worker-class gthread \
    --timeout 3600 \
    --graceful-timeout 120 \
    --keep-alive 5 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    "run:app"
