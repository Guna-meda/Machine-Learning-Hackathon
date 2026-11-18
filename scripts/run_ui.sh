#!/bin/bash
# Serve the evaluator UI on Laptop 2.  Then open http://localhost:8080/?api=http://<LAPTOP1_IP>:8000
cd "$(dirname "$0")/.."
python3 -m http.server 8080 --directory src/ui
