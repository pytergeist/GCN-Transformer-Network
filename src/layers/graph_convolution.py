import tensorflow as tf


class GraphConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.output_dim = output_dim
        self.dense = tf.keras.layers.Dense(output_dim, activation=None)

    def call(self, adjacency_matrix, node_features):
        degree_matrix_inv_sqrt = tf.linalg.diag(
            1.0 / tf.sqrt(tf.reduce_sum(adjacency_matrix, axis=-1))
        )
        normalized_adj = tf.matmul(
            tf.matmul(degree_matrix_inv_sqrt, adjacency_matrix), degree_matrix_inv_sqrt
        )
        aggregated_features = tf.matmul(normalized_adj, node_features)
        return tf.nn.relu(self.dense(aggregated_features))
