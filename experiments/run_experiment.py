import json
from itertools import product

from experiments.evaluate import evaluate_model
from experiments.preprocess import preprocess_data
from experiments.train import train_model


def run_experiments(ratings_path, trust_path, params_grid):
    X, A, ratings = preprocess_data(ratings_path, trust_path)

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.25, random_state=42)

    train_data = {
        "inputs": [train["user_idx"].values, train["item_idx"].values],
        "targets": train["rating"].values,
    }
    val_data = {
        "inputs": [val["user_idx"].values, val["item_idx"].values],
        "targets": val["rating"].values,
    }
    test_data = {
        "inputs": [test["user_idx"].values, test["item_idx"].values],
        "targets": test["rating"].values,
    }

    results = []
    for params in product(*params_grid.values()):
        current_params = dict(zip(params_grid.keys(), params))
        print(f"Running experiment with params: {current_params}")

        model, _ = train_model(X, A, train_data, val_data, current_params)
        metrics = evaluate_model(model, test_data)

        results.append(
            {"params": current_params, "mae": metrics[1], "rmse": metrics[2]}
        )

    with open("experiment_results.json", "w") as f:
        json.dump(results, f)


params_grid = {
    "gcn_layers": [1, 2],
    "embed_dim": [32, 64],
    "num_heads": [1, 2],
    "learning_rate": [0.001, 0.005],
    "batch_size": [64, 128],
    "epochs": [50],
    "dropout_rate": [0.1],
}

if __name__ == "__main__":
    run_experiments("ratings.csv", "trust.csv", params_grid)
