import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from mlp_regressor import MLPRegressor
from utils import split_dataset_into_train_val_test

PATHS = {
    "en": Path("data/hidden_states_hs_en.pkl"),
    "paper": Path("data/hidden_states_hs_paper.pkl"),
    "es": Path("data/hidden_states_hs_es.pkl"),
}
COUNTRIES = [
    "argentina",
    "chile",
    "colombia",
    "costa_rica",
    "cuba",
    "ecuador",
    "el_salvador",
    "guatemala",
    "honduras",
    "mexico",
    "nicaragua",
    "panama",
    "paraguay",
    "peru",
    "republica_dominicana",
    "usa",
    "venezuela",
]
PROJECT = "keen_probe_latam"
RESULTS_CSV = Path("results.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Carga un modelo entrenado y ejecuta el set de test asociado."
    )
    parser.add_argument("--prompt", choices=["es", "en", "paper"])
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_iter", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--country", choices=COUNTRIES)
    parser.add_argument("--region", choices=["latam", "usa"])
    args = parser.parse_args()

    if args.country and args.region:
        parser.error("--country y --region son mutuamente excluyentes")

    return args


def build_run_name(args) -> str:
    run_name = f"{PROJECT}_lr_{args.learning_rate}_epoch_{args.max_iter}_{args.prompt}"
    if args.country:
        run_name = f"{run_name}_{args.country}"
    elif args.region:
        run_name = f"{run_name}_{args.region}"
    return run_name


def load_test_split(args, device):
    df = pd.read_pickle(PATHS[args.prompt])
    _, _, _, _, X_test, y_test = split_dataset_into_train_val_test(
        df, country=args.country, region=args.region
    )

    X_test_np = np.array(X_test.tolist(), dtype=np.float32)
    y_test_np = np.array(y_test.tolist(), dtype=np.float32).reshape(-1, 1)

    X_test_tensor = torch.from_numpy(X_test_np).to(device)
    y_test_tensor = torch.from_numpy(y_test_np).to(device)
    return X_test_tensor, y_test_tensor


def evaluate(model: MLPRegressor, X_test: torch.Tensor, y_test: torch.Tensor):
    model.eval()
    with torch.no_grad():
        preds = model.predict(X_test.to(torch.float32))
        criterion = torch.nn.MSELoss()
        test_loss = criterion(preds, y_test.to(torch.float32)).item()

    result_df = pd.DataFrame(
        {
            "preds": preds.squeeze(dim=-1).detach().cpu().numpy(),
            "target": y_test.squeeze(dim=-1).detach().cpu().numpy(),
        }
    )
    test_pearson_corr, test_pearson_p_value = pearsonr(
        result_df["preds"], result_df["target"]
    )
    return test_loss, test_pearson_corr, test_pearson_p_value


def save_results(args, model_path: Path, test_loss: float, test_pearson_corr: float, test_pearson_p_value: float):
    record = {
        "prompt": args.prompt,
        "learning_rate": args.learning_rate,
        "epoch": args.max_iter,  # mismo valor que max_iter para mantener la etiqueta solicitada
        "max_iter": args.max_iter,
        "batch_size": args.batch_size,
        "country": args.country or "",
        "region": args.region or "",
        "model_path": str(model_path),
        "test_loss": test_loss,
        "test_pearson_corr": test_pearson_corr,
        "test_pearson_p_value": test_pearson_p_value,
    }
    fieldnames = list(record.keys())
    file_exists = RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = build_run_name(args)
    model_path = Path("probes") / f"{run_name}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en {model_path}. "
            "Asegúrate de haberlo entrenado con main.py."
        )

    with model_path.open("rb") as f:
        model: MLPRegressor = pickle.load(f)
    model.to(device)

    X_test, y_test = load_test_split(args, device)
    test_loss, test_pearson_corr, test_pearson_p_value = evaluate(
        model, X_test, y_test
    )

    print(f"Modelo: {model_path}")
    print(
        {
            "test_loss": test_loss,
            "test_pearson_corr": test_pearson_corr,
            "test_pearson_p_value": test_pearson_p_value,
        }
    )
    save_results(args, model_path, test_loss, test_pearson_corr, test_pearson_p_value)
    print(f"Resultados guardados en {RESULTS_CSV.resolve()}")


if __name__ == "__main__":
    main()
