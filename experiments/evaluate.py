def evaluate_model(model, test_data):
    results = model.evaluate(test_data["inputs"], test_data["targets"], batch_size=128)
    print(f"Test MAE: {results[1]}")
    print(f"Test RMSE: {results[2]}")
    return results
