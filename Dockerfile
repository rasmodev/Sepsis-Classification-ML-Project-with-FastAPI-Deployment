# Use the official Python image as a parent image
FROM python:3.11.3-slim

# Set the working directory within the container
WORKDIR /app

# Copy your FastAPI application code into the container
COPY ./src/main.py /app

# Copy the model and key components into the container
COPY ./model_and_key_components.pkl /app

# Copy the requirements.txt file into the container
COPY ./requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip install -r /app/requirements.txt

# Expose port 7860 for the FastAPI application
EXPOSE 7860

# Define the command to run your FastAPI application


