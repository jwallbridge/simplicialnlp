"""Implementation of multiheaded attention and self-attention layers.
Based on 
https://github.com/tensorflow/models/blob/master/official/transformer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
from keras import initializers
from keras.layers import Layer, activations
from tensorflow.keras import backend as K


class AttentionOne(tf.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, num_heads, num_vir_entities, attention_dropout, train):
    if hidden_size % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(AttentionOne, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_vir_entities = num_vir_entities
    self.attention_dropout = attention_dropout
    self.train = train

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
    self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
    self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

    self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                              name="output_transform")

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, hidden_size]
    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]
      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads)
      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth])
      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_size])

  def split_collapse(self, x):
    """Split and collapse bias term.
    Args:
      x: A tensor [batch_size, 1, 1, input_len]
    Returns"
      A tensor with shape [-1, 1, input_len]
    """  
    with tf.name_scope("split_collapse_bias"):
      identity = tf.shape(x)[2]
      length = tf.shape(x)[-1]
      x_tiled = K.tile(x, [1, self.num_heads, 1, 1])
      x = tf.reshape(x_tiled, (-1, identity, length))
      return x  

  def call(self, x, y, bias, cache=None):
    """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x + num_ve, hidden_size]
      y: a tensor with shape [batch_size, length_y + num_ve, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      AttentionOne layer output with shape [batch_size, length_x + num_ve, hidden_size]
    """
    num_ve = self.num_vir_entities
    length = tf.shape(x)[1] # input_length + num_ve
    depth = (self.hidden_size // self.num_heads)
    
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)
      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q) # shape (batch_size, num_heads, length, depth)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # collapse the batch dimension and head dimensions to operate simultaneously on all heads. 
    q = tf.reshape(q, (-1, length, depth)) # shape (-1, length, depth)
    k = tf.reshape(k, (-1, length, depth))
    v = tf.reshape(v, (-1, length, depth))
    
    # Scale q to prevent the dot product between q and k from growing too large.
    q *= depth ** -0.5

    # Calculate dot product attention. Only standard entities update reps of standard entities.
    # Virtual entities receive updates from all entities.
    logits_std = tf.einsum('aib,ajb->aij', q[:, :length-num_ve, :], k[:, :length-num_ve, :]) # (-1, len-num_ve, len-num_ve)
    logits_vir = tf.einsum('aib,ajb->aij', q[:, length-num_ve:, :], k) # (-1, num_ve, length)
    
    bias = self.split_collapse(bias)  # bias has shape (-1, 1, length)
    logits_std += bias  
   
    weights_std = tf.nn.softmax(logits_std, name="weights_qk_std")
    weights_vir = tf.nn.softmax(logits_vir, name="weights_qk_vir")
    
    if self.train:
      weights_std = tf.nn.dropout(weights_std, 1.0 - self.attention_dropout)
      weights_vir = tf.nn.dropout(weights_vir, 1.0 - self.attention_dropout)

    ao_std = tf.einsum('aij,ajc->aic', weights_std, v[:,:length-num_ve,:]) # shape (-1, length-num_ve, depth)
    ao_vir = tf.einsum('aij,ajc->aic', weights_vir, v) # shape (-1, num_ve, depth)
    ao = tf.concat([ao_std, ao_vir], axis=-2) # shape (-1, length, depth)
    
    attention_output = tf.reshape(ao, (-1, self.num_heads, length, depth))
    
    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    return attention_output  # shape (batch_size, length, hidden_size)

class SelfAttentionOne(AttentionOne):
  """Multiheaded self-attention layer."""
  def call(self, x, bias, cache=None):
    return super(SelfAttentionOne, self).call(x, x, bias, cache)    
