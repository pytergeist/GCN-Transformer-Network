import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(ratings_path="data/soc-Epinions1.txt", trust_path=None):
    """
    Preprocess the Epinions dataset and optional trust data.
    """
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating"],
        dtype={"user_id": str, "item_id": str},
        comment="#",  # Skip lines starting with '#'
    )

    print("Raw dataset preview:")
    print(ratings.head())

    if ratings["rating"].isnull().all():
        print("All ratings are NaN. Assigning default value of 1.0 to ratings.")
        ratings["rating"] = 1.0  # Default rating

    print("Ratings data after handling NaN:")
    print(ratings.head())

    users = ratings["user_id"].unique()
    items = ratings["item_id"].unique()

    user_mapping = {user: idx for idx, user in enumerate(users)}
    item_mapping = {item: idx for idx, item in enumerate(items)}

    print(f"Number of unique users: {len(users)}, Number of unique items: {len(items)}")

    ratings["user_idx"] = ratings["user_id"].map(user_mapping)
    ratings["item_idx"] = ratings["item_id"].map(item_mapping)

    print("Ratings data after mapping:")
    print(ratings.head())

    if ratings.empty:
        raise ValueError(
            "Dataset is empty after preprocessing. Please check the dataset format."
        )

    num_users = len(users)
    num_items = len(items)

    X = np.eye(num_users + num_items)

    A = None
    if trust_path:
        trust = pd.read_csv(
            trust_path,
            sep="\t",
            header=None,
            names=["user_id", "friend_id"],
            dtype={"user_id": str, "friend_id": str},
            comment="#",
        )
        trust["user_idx"] = trust["user_id"].map(user_mapping)
        trust["friend_idx"] = trust["friend_id"].map(user_mapping)

        print("Trust data preview:")
        print(trust.head())

        A = np.zeros((num_users, num_users))
        for _, row in trust.iterrows():
            if not np.isnan(row["user_idx"]) and not np.isnan(row["friend_idx"]):
                A[int(row["user_idx"]), int(row["friend_idx"])] = 1
                A[int(row["friend_idx"]), int(row["user_idx"])] = 1

    required_columns = ["user_idx", "item_idx", "rating"]
    train, test = train_test_split(
        ratings[required_columns], test_size=0.2, random_state=42
    )
    train, val = train_test_split(train, test_size=0.25, random_state=42)

    print("Train data preview:")
    print(train.head())

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

    return X, A, train_data, val_data, test_data


if __name__ == "__main__":
    X, A, train_data, val_data, test_data = preprocess_data()
    print(f"Train data: {train_data.keys()}")
    print(f"Train inputs: {train_data['inputs']}")
