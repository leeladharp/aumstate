#!/bin/bash

echo "Stopping AUM State..."

echo "Stopping Streamlit..."
pkill -f "streamlit run"

echo "Stopping Cloudflare tunnel..."
pkill -f cloudflared

echo "Stopping active Ollama models..."
ollama stop --all 2>/dev/null

echo "Checking remaining processes..."

ps aux | grep -E "streamlit|cloudflared|ollama" | grep -v grep

echo "AUM State stopped."
