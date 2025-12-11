import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
PATHS = {
    "en": {
        "score": Path("data/score.json"),
        "hidden_states": Path("data/hidden_states/hidden_states_en_hs.npz"),
        "output": Path("data/hidden_states_hs_en.pkl"),
    },
    "paper": {
        "score": Path("data/score.json"),
        "hidden_states": Path("data/hidden_states/hidden_states_paper_hs.npz"),
        "output": Path("data/hidden_states_hs_paper.pkl"),
    },
    "es": {
        "score": Path("data/score.json"),
        "hidden_states": Path("data/hidden_states/hidden_states_es_hs.npz"),
        "output": Path("data/hidden_states_hs_es.pkl"),
    }
}

def split_dataset_into_train_val_test(dataset, country=None, region=None):
    # Split in 65% train, 15% validation, 20% test
    if country:
        dataset = dataset[dataset["country"] == country]
    elif region:
        if region == "usa":
            dataset = dataset[dataset["country"] == "usa"]
        else:  # latam
            dataset = dataset[dataset["country"] != "usa"]
    X_train, X_temp, y_train, y_temp = train_test_split(dataset['hidden_states'], dataset['accuracy'], test_size=0.35, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=20/35, random_state=RANDOM_SEED)
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_hidden_states(npz_path: Path) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Load normalized representations and build an index that maps the vector_index
    to its position in the representations array.
    """
    npz_file = np.load(npz_path)
    representations = npz_file["representations_normalized"]
    row_indices = npz_file["row_indices"]
    index_lookup = {int(row_idx): position for position, row_idx in enumerate(row_indices)}
    return representations, index_lookup


def build_rows(
    score_path: Path, representations: np.ndarray, index_lookup: Dict[int, int]
) -> List[Dict[str, object]]:
    """
    Assemble the rows for the CSV by pairing each subject with its hidden state.
    """
    with score_path.open("r", encoding="utf-8") as score_file:
        scores = json.load(score_file)

    rows: List[Dict[str, object]] = []
    for entry in scores:
        vector_index = int(entry["vector_index"])
        representation_idx = index_lookup.get(vector_index)
        if representation_idx is None:
            raise KeyError(f"vector_index {vector_index} not found in row_indices")

        rows.append(
            {
                "subject": entry.get("entidad") or entry.get("subject"),
                "country": entry.get("pais") or entry.get("país") or entry.get("country"),
                "accuracy": entry.get("score"),
                "total_examples": entry.get("total_preguntas") or entry.get("total_examples"),
                # Keep the hidden state as a NumPy array for downstream processing.
                "hidden_states": representations[representation_idx].copy(),
            }
        )

    return rows


def write_pickle(rows: List[Dict[str, object]], output_path: Path) -> None:
    """
    Persist the rows as a pickled pandas DataFrame.

    Using pandas here ensures consumers can simply call pd.read_pickle(...)
    to get the DataFrame back.
    """
    df = pd.DataFrame(rows)
    with output_path.open("wb") as pkl_file:
        pickle.dump(df, pkl_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Procesa hidden states y genera un archivo PKL con los resultados."
    )
    parser.add_argument(
        "dataset",
        choices=list(PATHS.keys()),
        help=f"Selecciona el conjunto de datos a procesar. Opciones: {', '.join(PATHS.keys())}"
    )
    
    args = parser.parse_args()
    
    paths = PATHS[args.dataset]
    score_path = paths["score"]
    hidden_states_path = paths["hidden_states"]
    output_path = paths["output"]
    
    representations, index_lookup = load_hidden_states(hidden_states_path)
    rows = build_rows(score_path, representations, index_lookup)
    write_pickle(rows, output_path)
    
    print(f"Procesamiento completado: {args.dataset}")
    print(f"Archivo de salida: {output_path}")


if __name__ == "__main__":
    main()
