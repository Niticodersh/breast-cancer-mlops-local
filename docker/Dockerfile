# docker/Dockerfile
# Multi-stage Dockerfile for FastAPI inference service
# Stage 1: Build (install dependencies)
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for better caching)
COPY api/fastapi_app/requirements.txt .

# Install dependencies in a clean way
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime (small & secure final image)
FROM python:3.9-slim

WORKDIR /app

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy application code and model artifacts
COPY api/fastapi_app/main.py .
COPY model/ ./model/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose ports

##  FastAPI
EXPOSE 8000  
## Prometheus metrics  
EXPOSE 8001   

# Run the app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]