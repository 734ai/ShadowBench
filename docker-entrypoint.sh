#!/bin/bash
set -e

# ShadowBench Docker Entrypoint Script
echo "🛡️  Starting ShadowBench Enterprise Framework..."

# Initialize application
echo "📋 Initializing framework..."

# Create default configuration if not exists
if [ ! -f "config.yaml" ]; then
    echo "📝 Creating default configuration..."
    python shadowbench.py create-config -o config.yaml
fi

# Run based on command
case "$1" in
    "server")
        echo "🌐 Starting web dashboard..."
        python dashboard.py --port 8080
        ;;
    "benchmark")
        echo "⚡ Running performance benchmark..."
        python performance_benchmark.py
        ;;
    "test")
        echo "🧪 Running test suite..."
        python -m pytest test_framework.py -v
        ;;
    "cli")
        shift
        echo "💻 Running CLI command: $*"
        python shadowbench.py "$@"
        ;;
    *)
        echo "🚀 Running command: $*"
        exec "$@"
        ;;
esac
