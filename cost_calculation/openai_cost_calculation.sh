#!/bin/bash

# ==============================
# OpenAI Chat API + cost estimate
# Gets question from user, calls chat/completions, prints usage and cost.
# ==============================

set -e

# Script directory (so we find .env next to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load OPENAI_API_KEY from .env in script directory (strip optional quotes)
if [ -f "$SCRIPT_DIR/.env" ]; then
  _key_line=$(grep -E '^OPENAI_API_KEY=' "$SCRIPT_DIR/.env" | head -1)
  if [ -n "$_key_line" ]; then
    export OPENAI_API_KEY="${_key_line#OPENAI_API_KEY=}"
    export OPENAI_API_KEY="${OPENAI_API_KEY%\'}"; export OPENAI_API_KEY="${OPENAI_API_KEY#\'}"
    export OPENAI_API_KEY="${OPENAI_API_KEY%\"}"; export OPENAI_API_KEY="${OPENAI_API_KEY#\"}"
  fi
  _model_line=$(grep -E '^OPENAI_MODEL=' "$SCRIPT_DIR/.env" | head -1)
  if [ -n "$_model_line" ]; then
    export OPENAI_MODEL="${_model_line#OPENAI_MODEL=}"
    export OPENAI_MODEL="${OPENAI_MODEL%\'}"; export OPENAI_MODEL="${OPENAI_MODEL#\'}"
    export OPENAI_MODEL="${OPENAI_MODEL%\"}"; export OPENAI_MODEL="${OPENAI_MODEL#\"}"
  fi
fi

# ==============================
# Configuration
# ==============================

API_KEY="${OPENAI_API_KEY:-}"
MODEL="${OPENAI_MODEL:-gpt-4o-mini}"

# Pricing per 1K tokens (update based on actual model pricing; example: gpt-4o-mini)
INPUT_PRICE_PER_1K=0.00015
OUTPUT_PRICE_PER_1K=0.0006

# ==============================
# Validate
# ==============================

if [ -z "$API_KEY" ] || [ "$API_KEY" = "your-openai-api-key-here" ]; then
  echo "ERROR: Set OPENAI_API_KEY in $SCRIPT_DIR/.env or in the environment."
  echo "  Get a key at: https://platform.openai.com/account/api-keys"
  exit 1
fi

# Get question from user (or from first argument)
if [ -n "$1" ]; then
  PROMPT="$*"
else
  echo "Enter your question (or run with: $0 \"Your question here\"):"
  read -r PROMPT
fi

if [ -z "$PROMPT" ]; then
  echo "No prompt provided. Exiting."
  exit 1
fi

# Escape for JSON: replace \ and " and newlines
PROMPT_JSON=$(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

# ==============================
# API Call (Chat Completions)
# ==============================

response=$(curl -s "https://api.openai.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": $PROMPT_JSON}]
  }")

# Check for API errors
if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
  echo "API Error:"
  echo "$response" | jq -r '.error.message'
  exit 1
fi

# ==============================
# Extract content and token usage
# ==============================

content=$(echo "$response" | jq -r '.choices[0].message.content')
input_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens')
output_tokens=$(echo "$response" | jq -r '.usage.completion_tokens')
total_tokens=$(echo "$response" | jq -r '.usage.total_tokens')

# ==============================
# Cost calculation
# ==============================

input_cost=$(echo "scale=6; ($input_tokens / 1000) * $INPUT_PRICE_PER_1K" | bc)
output_cost=$(echo "scale=6; ($output_tokens / 1000) * $OUTPUT_PRICE_PER_1K" | bc)
total_cost=$(echo "scale=6; $input_cost + $output_cost" | bc)

# ==============================
# Output
# ==============================

echo ""
echo "=============================="
echo "Model: $MODEL"
echo "=============================="
echo "$content"
echo ""
echo "=============================="
echo "Token usage"
echo "=============================="
echo "Input tokens:   $input_tokens"
echo "Output tokens:  $output_tokens"
echo "Total tokens:   $total_tokens"
echo "------------------------------"
echo "Input cost:     \$$input_cost"
echo "Output cost:    \$$output_cost"
echo "Total cost:     \$$total_cost"
echo "=============================="
