# Use the official Python image as a parent image
FROM python:3.11.3

# Set environment variables for FastAPI host and port
ENV FASTAPI_HOST=0.0.0.0
ENV FASTAPI_PORT=8000

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the local directory to the working directory
COPY ./ /app

# Install any needed packages specified in requirements.txt
RUN pip install -r /app/requirements.txt

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Define the command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
