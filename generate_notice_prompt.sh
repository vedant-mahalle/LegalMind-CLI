#!/bin/bash

# Detailed Command Prompt for Legal Notice Generation
# This script prompts the user for all necessary details and generates a legal notice using main.py

echo "========================================"
echo "Legal Notice Generator - Command Prompt"
echo "========================================"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found in current directory."
    exit 1
fi

# Check if virtual environment is activated (optional check)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Make sure packages are installed."
fi

echo ""
echo "Step 1: Ensure vector store has data"
echo "If you haven't ingested PDFs yet, run:"
echo "python main.py ingest path/to/bare_act.pdf --label 'Bare Act Name'"
echo ""

read -p "Have you ingested PDFs into the vector store? (y/n): " has_data
if [ "$has_data" != "y" ] && [ "$has_data" != "Y" ]; then
    echo "Please ingest some PDFs first using: python main.py ingest <pdf_path>"
    exit 1
fi

echo ""
echo "Step 2: Configure the legal notice query"
echo ""

# Prompt for query
read -p "Enter the legal notice query (e.g., 'Draft a notice for breach of contract'): " query
if [ -z "$query" ]; then
    echo "Error: Query cannot be empty."
    exit 1
fi

# Prompt for number of context passages
read -p "Enter number of top-k context passages to retrieve (default 4): " k
k=${k:-4}
if ! [[ "$k" =~ ^[0-9]+$ ]]; then
    echo "Error: k must be a number."
    exit 1
fi

# Prompt for max tokens
read -p "Enter maximum tokens for Groq API response (default 4096): " max_tokens
max_tokens=${max_tokens:-4096}
if ! [[ "$max_tokens" =~ ^[0-9]+$ ]]; then
    echo "Error: max_tokens must be a number."
    exit 1
fi

# Prompt for output filename
read -p "Enter output PDF filename (leave empty for auto-generated): " output
output_option=""
if [ -n "$output" ]; then
    output_option="--output $output"
fi

# Prompt for including context
read -p "Include retrieved context as PDF appendix? (y/n, default y): " include_context
include_context=${include_context:-y}
if [ "$include_context" = "n" ] || [ "$include_context" = "N" ]; then
    include_context_option="--include-context false"
else
    include_context_option=""
fi

echo ""
echo "Step 3: Test Groq API connection (optional)"
read -p "Test Groq API connection first? (y/n): " test_api
if [ "$test_api" = "y" ] || [ "$test_api" = "Y" ]; then
    echo "Testing Groq API..."
    python main.py test_groq
    if [ $? -ne 0 ]; then
        echo "API test failed. Please check your .env file and API key."
        exit 1
    fi
    echo "API test successful!"
fi

echo ""
echo "Step 4: Generate the legal notice"
echo ""

# Construct the command
command="python main.py query \"$query\" --k $k --max-tokens $max_tokens $output_option $include_context_option"

echo "Executing command:"
echo "$command"
echo ""

# Execute the command
eval $command

echo ""
echo "========================================"
echo "Legal Notice Generation Complete!"
echo "========================================"