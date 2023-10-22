# Build the Docker container for the FastAPI application
docker build -t sepsis_fastapi -f Dockerfile .

# List all Docker images
docker images

# Run the Docker container locally
docker run -p 8000:8000 --name sepsis_fastapi dab03c5810df

# List running Docker containers
docker ps