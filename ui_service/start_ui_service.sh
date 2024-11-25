#!/bin/bash

# Configuration
NETWORK_NAME="nanoporous-network"  # Network to connect to the database service
UI_IMAGE_NAME="nanoporous-ui-image"
UI_CONTAINER_NAME="nanoporous-ui-service"
UI_PORT=8501  # Port for the UI service

# # Step 1: Ensure Docker network exists
# if ! docker network ls | grep -q $NETWORK_NAME; then
#   echo "Error: Docker network $NETWORK_NAME does not exist."
#   echo "Please start the database service with the same network."
#   exit 1
# else
#   echo "Docker network $NETWORK_NAME found."
# fi

# Step 2: Build the UI service image (if not already built)
echo "Building UI service image..."
docker build -t $UI_IMAGE_NAME .
if [ $? -ne 0 ]; then
  echo "Failed to build the UI service image."
  exit 1
fi
echo "UI service image built successfully."

# Step 3: Stop and remove any existing container with the same name
if [ "$(docker ps -aq -f name=$UI_CONTAINER_NAME)" ]; then
  echo "Stopping and removing existing UI service container..."
  docker stop $UI_CONTAINER_NAME
  docker rm $UI_CONTAINER_NAME
fi

# Step 4: Start the UI service
echo "Starting UI service..."
docker run -d \
  --name $UI_CONTAINER_NAME \
  --network $NETWORK_NAME \
  -p $UI_PORT:8501 \
  $UI_IMAGE_NAME

if [ $? -ne 0 ]; then
  echo "Failed to start the UI service."
  exit 1
fi

echo "UI service is running at: http://localhost:$UI_PORT"
