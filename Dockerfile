# Use Python 3.11 slim which has great compatibility with PyTorch and OpenCV
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements list
COPY requirements.txt .

# Install Python dependencies (this will take a few minutes in Cloud Build)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Cloud Run default port
ENV PORT 8080
EXPOSE $PORT

# Command to run the application using Uvicorn
# Binds to 0.0.0.0 and dynamically grabs the $PORT environment variable
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
