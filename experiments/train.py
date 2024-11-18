import tensorflow as tf
from src.models.graph_transformer_network import GraphTransformerNetwork


def train_model(X, A, train_data, val_data, params):
    model = GraphTransformerNetwork(
        num_gcn_layers=params["gcn_layers"],
        gcn_output_dim=params["embed_dim"],
        transformer_layers=params["transformer_layers"],
        transformer_embed_dim=params["embed_dim"],
        transformer_num_heads=params["num_heads"],
        transformer_ff_dim=params["embed_dim"] * 4,
        transformer_dropout_rate=params["dropout_rate"],
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="mean_squared_error",
        metrics=["mean_absolute_error", tf.keras.metrics.RootMeanSquaredError()],
    )

    history = model.fit(
        [X, A],
        train_data["targets"],
        validation_data=([val_data["inputs"], val_data["targets"]]),
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
    )

    return model, history
