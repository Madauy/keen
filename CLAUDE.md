# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KEEN (Knowledge Estimation in Neural Networks) implementation for measuring LLM knowledge about entities without generating tokens. This project evaluates whether language models have more knowledge about US entities versus Latin American entities through three stages: validation, extraction, and probe training.

## Repository Structure

The project is organized as three independent modules, each with its own `requirements.txt`:

```plaintext
keen/
├── extractor/    # Hidden states extraction from Falcon3-3B-Instruct
├── probe/        # MLP probe training on extracted representations
└── validation/   # LLM Q&A evaluation with semantic similarity
```

## Commands

### Extractor (Hidden States Extraction)

```bash
cd extractor
pip install -r requirements.txt

# Extract hidden states using different methods
python main.py --data ./questions.csv --method hs              # Hidden States (3072-D)
python main.py --data ./questions.csv --method vp-k --k 50     # Top-k Vocab Projection
python main.py --data ./questions.csv --country chile,usa --limit 100

# Resume interrupted extraction (same command continues from saved state)
python main.py --data ./questions.csv --method hs

# Apply normalization after complete extraction
python main.py --data ./questions.csv --normalize-only
```

### Probe (MLP Training)

```bash
cd probe

# Generate DataFrame from hidden states + scores
python utils.py paper  # or: en, es

# Train probe
python main.py --prompt paper --learning_rate 0.01 --max_iter 100 --batch_size 32
python main.py --prompt paper --country chile  # Filter by country
python main.py --prompt paper --region latam   # Filter by region (latam/usa)
```

### Validation (LLM Evaluation)

```bash
cd validation
pip install -r requirements.txt

python main.py --data ./questions.csv
python main.py --data ./questions.csv --model "Qwen/Qwen3-4B-Instruct" --language en
python main.py --data ./questions.csv --limit 100 --country chile,usa
```

### Linting (all modules)

```bash
ruff check .
ruff format .
```

## Architecture

### Pipeline Flow

1. **Validation**: Evaluate LLM on Q&A tasks to compute accuracy scores per entity
2. **Extraction**: Extract hidden states from entity tokens (layers 15-17 of Falcon3-3B)
3. **Probe**: Train MLP regressor to predict accuracy from hidden states (Pearson correlation)

### Extraction Methods (extractor/extractor.py)

- `hs`: Hidden States - 3072-D vectors from layers 15-17
- `vp`: Vocabulary Projection - 131K-D full vocabulary projection
- `vp-k`: Top-k Vocabulary Projection - 3k-D interpretable vectors

### State Management

All modules support resumable execution via state files:

- Extractor: `states/state_{hash}_{method}.json` + `.npz`
- Validation: `states/state_{hash}.json`

### CSV Input Format

```plaintext
entidad,relacion,objetos,pregunta,respuestas,obtenido_de,respuestas_aliases
```

The `obtenido_de` field contains country codes used for filtering.

## Code Style

- 2-space indentation (configured in ruff.toml)
- 100-character line length
- Ruff rules: E, F, I (pycodestyle, pyflakes, isort)
