#!/bin/bash
echo "Starting HTTP server..."
exec python -m http.server 8083 --directory /reports
