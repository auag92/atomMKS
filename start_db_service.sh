#!/bin/bash

# Configuration variables
IMAGE_NAME="nanoporous-image"
CONTAINER_NAME="nanoporous-service"
PORT=8000
DB_PATH="$(pwd)/persist"  # Dynamically set DB path based on the current working directory
CPUS=8
MEMORY="16g"
DOCKERFILE="Dockerfile"  # Specify the Dockerfile name

# Check if the persist directory exists
if [ ! -d "$DB_PATH" ]; then
    echo "Error: Directory 'persist' does not exist in the current directory."
    echo "Please create the directory to persist the database."
    exit 1
fi

# Step 1: Build the Docker image
echo "Building the Docker image..."
docker build -t $IMAGE_NAME -f $DOCKERFILE .
if [ $? -ne 0 ]; then
    echo "Error: Failed to build the Docker image."
    exit 1
fi
echo "Docker image '$IMAGE_NAME' built successfully."

# Step 2: Stop and remove any existing container with the same name
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Step 3: Run the Docker container
echo "Starting the Docker container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8000 \
    -v $DB_PATH:/data \
    --cpus=$CPUS \
    --memory=$MEMORY \
    --network nanoporous-network \
    $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo "Service started and running on port $PORT"
else
    echo "Error: Failed to start the Docker container."
    exit 1
fi
