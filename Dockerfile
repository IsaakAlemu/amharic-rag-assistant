# Lightweight production container for Amharic RAG Assistant
FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir google-genai

# Copy application source code
COPY . .

# Expose default Streamlit port
EXPOSE 8501

# Healthcheck for container stability
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit web service
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
