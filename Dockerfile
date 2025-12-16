# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for MoviePy, ffmpeg, git
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Expose the port Render assigns
EXPOSE $PORT

# Start the FastAPI app using Uvicorn and Render's dynamic port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
