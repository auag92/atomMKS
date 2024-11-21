# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    nano \
    git \
    sqlite3 \
    htop \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file (if any) for Python dependencies
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set environment variable for SQLite database path
ENV DATABASE_URL=sqlite:////data/materials.db

# Copy the rest of the application code into the container
COPY . /app

# Install your Python package (if applicable)
RUN pip install .

# # Set default entry point
# CMD ["bash"]

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the app with Uvicorn
CMD ["uvicorn", "services.n2dm_app:n2dm_app", "--host", "0.0.0.0", "--port", "8000"]
