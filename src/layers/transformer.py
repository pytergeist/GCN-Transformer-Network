from src.layers import MultiHeadSelfAttention
import tensorflow as tf


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.feed_forward = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(feed_forward_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
        attn_output = self.self_attention(x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)
        feed_forward_output = self.feed_forward(out1)
        feed_forward_output = self.dropout2(feed_forward_output, training=training)
        return self.norm2(out1 + feed_forward_output)
