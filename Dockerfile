# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for lxml and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set the working directory to ui folder
WORKDIR /app/ui

# Default port
ENV PORT=8501

# Expose port
EXPOSE 8501

# Use ENTRYPOINT with shell to handle $PORT
ENTRYPOINT ["/bin/sh", "-c", "exec streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]

