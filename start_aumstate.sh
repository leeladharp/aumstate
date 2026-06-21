#!/bin/bash

echo "Starting AUM State..."

cd ~/projects/aumstate || exit 1

source venv/bin/activate

echo "Checking Ollama..."
ollama list >/dev/null 2>&1

echo "Starting Streamlit..."
nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &

sleep 5

echo "Starting Cloudflare tunnel..."
nohup cloudflared tunnel run aumstate > cloudflared.log 2>&1 &

echo "AUM State started."
echo "Local: http://localhost:8501"
echo "Web:   https://app.aumstate.com"
