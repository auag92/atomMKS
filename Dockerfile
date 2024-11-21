# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    nano \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file (if any) for Python dependencies
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Install your Python package (if applicable)
RUN pip install .

# Set default entry point
CMD ["bash"]
