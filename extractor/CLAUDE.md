# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a KEEN (Knowledge Estimation in Neural Networks) implementation for extracting internal representations from Falcon3-3B-Instruct to measure LLM knowledge about entities without generating tokens. The project compares knowledge about Latin American vs. US entities.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run extraction CLI
python main.py --data ./questions.csv --method hs
python main.py --data ./questions.csv --method vp-k --k 50 --country chile,usa
python main.py --data ./questions.csv --limit 100  # Process first 100 pending

# Resume interrupted extraction (same command continues from saved state)
python main.py --data ./questions.csv --method hs

# Normalize after complete extraction
python main.py --data ./questions.csv --normalize-only

# Lint with ruff
ruff check .
ruff format .
```

## Architecture

### Core Components

- **`extractor.py`**: `FalconHiddenStatesExtractor` class implementing three KEEN methods:
  - `method_hs()`: Hidden States (3072-D vectors)
  - `method_vp()`: Vocabulary Projection (131K-D vectors)
  - `method_vp_k()`: Top-k Vocabulary Projection (3k-D vectors, interpretable)

- **`state_manager.py`**: `ExtractionStateManager` for resumable extraction with JSON state + NPZ vector storage

- **`main.py`**: CLI entry point with argparse

- **`config.py`**: Centralized configuration (model, CSV columns, prompt templates ES/EN)

### Data Flow

1. CSV with entities → `main.py` validates structure
2. For each entity: prompt "Dime todo lo que sabes de {entidad}" → forward pass (no generation)
3. Extract hidden states from entity token position at layers 15-17 (3/4 of model depth)
4. Buffer 10 rows → flush to NPZ file
5. When complete: apply global MinMax normalization across all vectors

### State Files

State files are stored in `states/` directory:

- `state_{hash}_{method}.json`: Metadata and row tracking
- `state_{hash}_{method}.npz`: Raw and normalized vectors

## Code Style

- 2-space indentation (configured in ruff.toml)
- Line length: 100 characters
- Ruff linting: E, F, I rules enabled
