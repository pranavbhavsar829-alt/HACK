#!/bin/bash

# Start the fetcher (bot logic) in the background
python fetcher.py &

# Start the web server (dashboard) in the foreground
gunicorn server:app --bind 0.0.0.0:$PORT