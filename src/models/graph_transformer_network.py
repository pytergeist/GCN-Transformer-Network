import tensorflow as tf

from src.layers import GraphConvolutionLayer, TransformerLayer


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
