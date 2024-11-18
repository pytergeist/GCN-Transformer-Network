import tensorflow as tf


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = embed_dim // num_heads
        self.q_dense = tf.keras.layers.Dense(embed_dim)
        self.k_dense = tf.keras.layers.Dense(embed_dim)
        self.v_dense = tf.keras.layers.Dense(embed_dim)
        self.out_dense = tf.keras.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        q = self.split_heads(self.q_dense(x), batch_size)
        k = self.split_heads(self.k_dense(x), batch_size)
        v = self.split_heads(self.v_dense(x), batch_size)

        attention_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.depth, tf.float32)
        )
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, v)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention_output, (batch_size, -1, self.embed_dim)
        )
        return self.out_dense(concat_attention)
