from src.layers import GraphConvolutionLayer, TransformerLayer
import tensorflow as tf


class GraphTransformerNetwork(tf.keras.Model):
    def __init__(
        self,
        num_gcn_layers,
        gcn_output_dim,
        transformer_layers,
        transformer_embed_dim,
        transformer_num_heads,
        transformer_ff_dim,
        transformer_dropout_rate,
    ):
        super(GraphTransformerNetwork, self).__init__()
        self.gcn_layers = [
            GraphConvolutionLayer(gcn_output_dim) for _ in range(num_gcn_layers)
        ]
        self.transformer_layers = [
            TransformerLayer(
                transformer_embed_dim,
                transformer_num_heads,
                transformer_ff_dim,
                transformer_dropout_rate,
            )
            for _ in range(transformer_layers)
        ]
        self.final_dense = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, adjacency_matrix, node_features, training=False):
        x = node_features
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(adjacency_matrix, x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training)
        return self.final_dense(x)


if __name__ == "__main__":
    num_nodes = 500
    num_features = 128
    adjacency_matrix = tf.random.uniform((num_nodes, num_nodes), minval=0, maxval=1)
    node_features = tf.random.uniform((num_nodes, num_features))

    model = GraphTransformerNetwork(
        num_gcn_layers=2,
        gcn_output_dim=64,
        transformer_layers=2,
        transformer_embed_dim=64,
        transformer_num_heads=4,
        transformer_ff_dim=256,
        transformer_dropout_rate=0.1,
    )

    output = model(adjacency_matrix, node_features)
    print(output.shape)
