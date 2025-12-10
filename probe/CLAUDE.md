# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

MLP probe training module for the KEEN project. Trains a single-layer linear regressor to predict entity accuracy scores from hidden state representations extracted from Falcon3-3B-Instruct. Uses Pearson correlation as the evaluation metric.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate DataFrame from hidden states + scores (required before training)
python utils.py paper  # Options: en, es, paper

# Train probe with hyperparameters
python main.py --prompt paper --learning_rate 0.01 --max_iter 100 --batch_size 32

# Filter by country or region
python main.py --prompt paper --country chile --learning_rate 0.01 --max_iter 100 --batch_size 32
python main.py --prompt paper --region latam --learning_rate 0.01 --max_iter 100 --batch_size 32

# Run all country experiments
bash run_all_countries.sh

# Linting
ruff check .
ruff format .
```

## Architecture

### Data Flow

1. **utils.py**: Combines `data/score.json` + `data/hidden_states/*.npz` → `data/hidden_states_hs_*.pkl`
2. **main.py**: Loads `.pkl`, splits 65/15/20 train/val/test, trains MLPRegressor
3. **probes/**: Trained models saved as `.pkl` files

### Key Components

- **MLPRegressor** (`mlp_regressor.py`): Single linear layer + sigmoid activation. Uses MSE loss and AdamW optimizer. Tracks best weights by validation Pearson correlation. Based on original KEEN code with wandb logging removed.

- **HiddenStatesDataset** (`main.py:57`): PyTorch Dataset wrapping the DataFrame for DataLoader.

### Data Format

- **Input `.npz`**: Keys `representations_normalized` (3072-D vectors) and `row_indices` (mapping to score.json)
- **score.json**: Entity metadata with `vector_index`, `score`, `pais/country`, `total_preguntas`
- **Output `.pkl`**: DataFrame with columns: `subject`, `country`, `accuracy`, `total_examples`, `hidden_states`

### Filtering

- `--country`: Single country (e.g., `chile`, `usa`, `mexico`)
- `--region`: `latam` (all non-USA) or `usa`
- Mutually exclusive options

### Countries

argentina, chile, colombia, costa_rica, cuba, ecuador, el_salvador, guatemala, honduras, mexico, nicaragua, panama, paraguay, peru, republica_dominicana, usa, venezuela

## Code Style

- 2-space indentation
- 100-character line length
- Ruff rules: E, F, I (pycodestyle, pyflakes, isort)
- Deterministic execution: seed 42, CUDA deterministic flags enabled
