FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first to leverage Docker cache
COPY prototype/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the common code (models, utils, checkpoints)
COPY models /app/models
COPY utils /app/utils
COPY checkpoints /app/checkpoints

# Copy the backend code
COPY prototype/backend /app/prototype/backend

# Ensure prototype is a package
RUN touch /app/prototype/__init__.py

# Expose the port
EXPOSE 8000

# Run the FastAPI server
CMD ["python", "-m", "prototype.backend.main"]
