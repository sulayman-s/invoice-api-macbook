# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/img

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the /app directory and its subdirectories are writable
RUN chmod -R 755 /app

# Ensure python-dotenv is installed
RUN pip install python-dotenv

# Copy the .env file
COPY .env /app/.env

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the command to start the server
CMD ["uvicorn", "invoice_data_processing.main:app", "--host", "0.0.0.0", "--port", "8000"]
