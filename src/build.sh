# Build the Docker container for the FastAPI application
docker build -t sepsis_fastapi src/Dockerfile .

# List all Docker images
docker images

# Run the Docker container locally
docker run -p 8000:8000 --name sepsis_fastapi "add the docker image that will be generated"

# List running Docker containers
docker ps