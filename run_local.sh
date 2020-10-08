#!/bin/bash

set -e

# gunicorn --workers 4 --worker-class gevent --bind 0.0.0.0:8050 --timeout 120 app:server
python3 app.py
