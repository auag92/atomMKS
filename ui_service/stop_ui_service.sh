#!/bin/bash

# Configuration variables
CONTAINER_NAME="nanoporous-ui-service"

# Stop and remove the container
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo "Service stopped and container removed"
