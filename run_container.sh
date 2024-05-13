#!/bin/bash

# Build Docker image
docker build -t intent-classifier-app .

# Run Docker container
docker run -p 8080:8080 intent-classifier-app

## run chmod +x run_container.sh, to make the script executable 
## run ./run_container.sh, to build and run the app


