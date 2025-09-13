#!/usr/bin/env python3
"""Test MLX directly without server."""

import mlx_lm

# Load model
model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Test generation
prompt = "Rephrase this to be more systematic: To be productive you need to work hard."
response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=50)

print(f"Prompt: {prompt}")
print(f"Response: {response}")
