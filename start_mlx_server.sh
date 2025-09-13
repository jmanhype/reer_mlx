#!/bin/bash
# Start MLX-LM server for DSPy

echo "Starting MLX-LM server on port 11434..."
echo "Model: berkeley-nest/Starling-LM-7B-alpha"
echo ""
echo "This will download the model if not cached (~14GB)"
echo "Press Ctrl+C to stop the server"
echo ""

python -m mlx_lm.server \
    --model mlx-community/Starling-LM-7B-alpha-4bit \
    --port 11434 \
    --max-tokens 2048 \
    --temp 0.7