def train_model(X, A, train_data, val_data, params):
    import tensorflow as tf

    from src.models.graph_transformer_network import GraphTransformerNetwork

    model = GraphTransformerNetwork(
        num_gcn_layers=params["gcn_layers"],
        gcn_output_dim=params["embed_dim"],
        transformer_layers=params["transformer_layers"],
        transformer_embed_dim=params["embed_dim"],
        transformer_num_heads=params["num_heads"],
        transformer_ff_dim=params["embed_dim"] * 4,
        transformer_dropout_rate=params["dropout_rate"],
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    rmse_metric = tf.keras.metrics.RootMeanSquaredError()

    adjacency_matrix = A
    node_features = X

    train_user_indices = train_data["inputs"][0]
    train_item_indices = train_data["inputs"][1]

    num_users = len(set(train_user_indices))
    node_indices = train_user_indices

    epochs = params["epochs"]
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            outputs = model(adjacency_matrix, node_features, training=True)
            predictions = tf.gather(outputs, node_indices)
            loss = mse_loss_fn(train_data["targets"], predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        mae_metric.update_state(train_data["targets"], predictions)
        rmse_metric.update_state(train_data["targets"], predictions)
        mae = mae_metric.result().numpy()
        rmse = rmse_metric.result().numpy()
        mae_metric.reset_states()
        rmse_metric.reset_states()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}, MAE: {mae}, RMSE: {rmse}"
        )

    return model, None
