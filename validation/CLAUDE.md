# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Validation is a resumable LLM validation system that evaluates language models on question-answering tasks. It processes questions from CSV files, generates responses using HuggingFace models, and evaluates correctness using semantic similarity.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run validation
python main.py --data /path/to/data.csv

# With options
python main.py --data ./questions.csv --model "Qwen/Qwen3-4B-Instruct" --language en --limit 100 --country chile,usa

# Lint code
ruff check .
ruff format .
```

## Architecture

The system has three main components:

- **main.py**: CLI entry point. Parses arguments, orchestrates the validation loop, and handles interrupts gracefully
- **evaluator.py**: Contains `Evaluator` (LLM inference) and `SimilarityEvaluator` (answer comparison using sentence-transformers)
- **state_manager.py**: Handles resumable execution via JSON state files. State files are named `state_{hash}.json` based on the input CSV path hash

### Data Flow

1. StateManager loads/creates state from CSV (columns: entidad, relacion, objetos, pregunta, respuestas, obtenido_de, respuestas_aliases)
2. Evaluator generates responses using chat template format with few-shot examples
3. SimilarityEvaluator compares responses using exact match first, then semantic similarity (threshold: 0.80). Years require exact match only.
4. Results are saved immediately after each question (atomic JSON writes)

### Key Configuration (config.py)

- Default model: `tiiuae/Falcon3-3B-Instruct`
- Similarity model: `sentence-transformers/distiluse-base-multilingual-cased-v2`
- Supported languages: `es` (Spanish), `en` (English)

## Code Style

Uses ruff with 2-space indentation and 100-char line length. Lint rules: E, F, I (pycodestyle errors, pyflakes, isort).
