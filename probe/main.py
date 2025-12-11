import argparse
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from mlp_regressor import MLPRegressor
from utils import split_dataset_into_train_val_test

PATHS = {
    "en": Path("data/hidden_states_hs_en.pkl"),
    "paper": Path("data/hidden_states_hs_paper.pkl"),
    "es": Path("data/hidden_states_hs_es.pkl")
}
COUNTRIES = ["argentina", "chile", "colombia", "costa_rica", "cuba", "ecuador", "el_salvador", "guatemala", "honduras", "mexico", "nicaragua", "panama", "paraguay", "peru", "republica_dominicana", "usa", "venezuela"]

class HiddenStatesDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.df = X_train
        self.labels = y_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        hidden_states = self.df.iloc[idx]
        accuracy = self.labels.iloc[idx]
        return torch.tensor(hidden_states, dtype=torch.float32), torch.tensor(accuracy, dtype=torch.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=["es", "en", "paper"]) 
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_iter", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--country", choices=COUNTRIES)
    parser.add_argument("--region", choices=["latam", "usa"])

    args = parser.parse_args()

    if args.country and args.region:
        parser.error("--country y --region son mutuamente excluyentes")
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    df = pd.read_pickle(PATHS[args.prompt])
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_into_train_val_test(df, country=args.country, region=args.region)
    dataset = HiddenStatesDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    input_size = len(dataset[0][0])
    
    classifier_model_params = {
        "input_size": input_size,
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "max_iter": args.max_iter,
    }
    project = f"keen_probe_latam"
    run_name = f"{project}_lr_{classifier_model_params['learning_rate']}_epoch_{classifier_model_params['max_iter']}_{args.prompt}"
    if args.country:
        run_name = f"{run_name}_{args.country}"
    elif args.region:
        run_name = f"{run_name}_{args.region}"

    # Build the MLPRegressor model and train it
    model = MLPRegressor(**classifier_model_params).to(device)
    model.fit(dataloader, y_train, X_val, y_val) 
    with open(f"probes/{run_name}_model.pkl",'wb') as f:
        model.set_to_best_weights()
        pickle.dump(model, f)