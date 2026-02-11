#!/bin/bash
# entrypoint.sh
set -e

echo "🚀 Запускаем сервер..."
ollama serve &
PID=$!
sleep 15

echo "Checking if model exists..."
if ! ollama list | grep -q "qwen2.5:7b"; then
    echo "Downloading Qwen 14B model (q4_K_M)..."
    ollama pull qwen2.5:14b
else
    echo "Model already exists."
fi

wait $PID