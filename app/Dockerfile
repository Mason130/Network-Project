# Use an official Python runtime as a parent image
# FROM ubuntu:20.04
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install packages
RUN apt-get update && \
    apt-get install -y \
    iptables \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
# Run app.py when the container launches
# CMD ["python", "app.py"]
CMD ["python", "app_protected.py"]
