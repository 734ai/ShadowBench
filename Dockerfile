# ShadowBench Enterprise - Production Docker Image
FROM python:3.13-slim

# Metadata
LABEL maintainer="ShadowBench Team"
LABEL description="Enterprise AI Security Benchmarking Framework"
LABEL version="1.0.0-beta"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r shadowbench && useradd -r -g shadowbench shadowbench

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs results data && \
    chown -R shadowbench:shadowbench /app

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER shadowbench

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python shadowbench.py version || exit 1

# Expose ports
EXPOSE 8080 8000

# Default command
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["server"]
