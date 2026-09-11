FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install system build dependencies required for ML libraries (libgomp for xgboost/lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Hugging Face Spaces runs on non-root user (UID 1000) and port 7860
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

EXPOSE 7860

# Run with Gunicorn dynamically on $PORT provided by host (Render uses 10000 or custom)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --timeout 120 --workers 2 app:app"]
