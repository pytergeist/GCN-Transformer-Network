import numpy as np
import pandas as pd


def preprocess_data(ratings_path, trust_path):

    ratings = pd.read_csv(ratings_path)
    users = ratings["user_id"].unique()
    items = ratings["item_id"].unique()

    user_mapping = {user: idx for idx, user in enumerate(users)}
    item_mapping = {item: idx for idx, item in enumerate(items)}

    ratings["user_idx"] = ratings["user_id"].map(user_mapping)
    ratings["item_idx"] = ratings["item_id"].map(item_mapping)

    num_users = len(users)
    num_items = len(items)
    X = np.zeros((num_users + num_items, num_users + num_items))
    for _, row in ratings.iterrows():
        user_idx = row["user_idx"]
        item_idx = row["item_idx"] + num_users
        X[user_idx, item_idx] = 1
        X[item_idx, user_idx] = 1

    trust = pd.read_csv(trust_path)
    trust["user_idx"] = trust["user_id"].map(user_mapping)
    trust["friend_idx"] = trust["friend_id"].map(user_mapping)
    A = np.zeros((num_users, num_users))
    for _, row in trust.iterrows():
        A[row["user_idx"], row["friend_idx"]] = 1
        A[row["friend_idx"], row["user_idx"]] = 1

    return X, A, ratings[["user_idx", "item_idx", "rating"]]
