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

def multi_softmax(target, axis, name=None):
  with tf.name_scope(name, 'softmax', values=[target]):
    max_axis = tf.reduce_max(target, axis, keep_dims=True)
    target_exp = tf.exp(target-max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
    softmax = target_exp / normalize
    return softmax

class AttentionThree(tf.layers.Layer):
  """Multi-headed 3-simplex attention layer."""

  def __init__(self, hidden_size, d3_model: int, num_heads, num_vir_entities1: int, num_vir_entities2: int, 
                attention_dropout, train):
    if d3_model % num_heads != 0:
      raise ValueError("Three simplex dimension must be evenly divisible by the number of "
                       "heads.")

    super(AttentionThree, self).__init__()
    self.hidden_size = hidden_size
    self.d3_model = d3_model
    self.num_heads = num_heads
    self.num_vir_entities1 = num_vir_entities1
    self.num_vir_entities2 = num_vir_entities2
    self.attention_dropout = attention_dropout
    self.train = train

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.layers.Dense(d3_model, use_bias=False, name="q")
    self.k1_dense_layer = tf.layers.Dense(d3_model, use_bias=False, name="k1")
    self.k2_dense_layer = tf.layers.Dense(d3_model, use_bias=False, name="k2")
    self.k3_dense_layer = tf.layers.Dense(d3_model, use_bias=False, name="k3")
    self.v_dense_layer = tf.layers.Dense(d3_model, use_bias=False, name="v")

    self.output_dense_layer = tf.layers.Dense(d3_model, use_bias=False,
                                              name="output_transform3")

  def build(self, input_shape):
    self.q_weights = tf.get_variable("q", 
           shape=[self.hidden_size, self.d3_model], initializer=None, trainable=True)
    self.k1_weights = tf.get_variable("k1", 
           shape=[self.hidden_size, self.d3_model], initializer=None, trainable=True)
    self.k2_weights = tf.get_variable("k2", 
           shape=[self.hidden_size, self.d3_model], initializer=None, trainable=True)                                       
    self.k3_weights = tf.get_variable("k3", 
           shape=[self.hidden_size, self.d3_model], initializer=None, trainable=True)                                       
    self.v_weights = tf.get_variable("v", 
           shape=[self.hidden_size, self.d3_model], initializer=None, trainable=True)
    self.B_weights = tf.get_variable("C",
            shape=[self.d3_model // self.num_heads, self.d3_model // self.num_heads,
                   self.d3_model // self.num_heads, self.d3_model // self.num_heads], trainable=True) 

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, d_model]
    Returns:
      A tensor with shape [batch_size, num_heads, length, d_model/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth3 = (self.d3_model // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth3])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, d_model/num_heads]
    Returns:
      A tensor with shape [batch_size, length, d_model]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth3]
      return tf.reshape(x, [batch_size, length, self.d3_model])

  def call(self, x, y, bias, cache=None):
    """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, d3_model]
      y: a tensor with shape [batch_size, length_y, d3_model]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results    CHECK !!!!
              of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      AttentionThree layer output with shape [batch_size, length_x, d3_model]
    """
    num_ve1 = self.num_vir_entities1  # virtual entities for 2-simplices
    num_ve2 = self.num_vir_entities2  # virtual entities for 3-simplices
    length = tf.shape(x)[1]  # input_length + num_ve
    depth3 = (self.d3_model // self.num_heads)
    
    # Linearly project the query (q), key one (k1), key two (k2), key three (k3) and value (v) 
    # using different learned projections. 
    q = K.dot(x, self.q_weights) # shape (batch_size, length, d3_model)
    k1 = K.dot(y, self.k1_weights)
    k2 = K.dot(y, self.k2_weights)
    k3 = K.dot(y, self.k3_weights)
    v = K.dot(y, self.v_weights)
    
    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k1 = tf.concat([cache["k1"], k1], axis=1)
      k2 = tf.concat([cache["k2"], k2], axis=1)
      k3 = tf.concat([cache["k3"], k3], axis=1)
      v = tf.concat([cache["v"], v], axis=1)
      # Update cache
      cache["k1"] = k1
      cache["k2"] = k2
      cache["k3"] = k3
      cache["v"] = v

    # Split q, k1, k2, k3, v into heads.
    q = self.split_heads(q)
    k1 = self.split_heads(k1)
    k2 = self.split_heads(k2)
    k3 = self.split_heads(k3)
    v = self.split_heads(v)
    
    v_virtual_orig = v[:,:,length-num_ve1-num_ve2:,:] # shape (batch_size, num_heads, length-num_ve1-num_ve2, depth3)
    
    # collapse the batch dimension and head dimensions to operate simultaneously on all heads. 
    q = tf.reshape(q, (-1, length, depth3)) # shape (-1, length, depth3)
    k1 = tf.reshape(k1, (-1, length, depth3))
    k2 = tf.reshape(k2, (-1, length, depth3))
    k3 = tf.reshape(k3, (-1, length, depth3))
    v = tf.reshape(v, (-1, length, depth3))

    # We generate queries only for standard entities, and keys and
    # values only for virtual entities
    q = q[:, :length-num_ve1-num_ve2, :] # shape (batch_size, length-num_ve1-num_ve2, d3_model)
    k1 = k1[:, length-num_ve2:, :] 
    k2 = k2[:, length-num_ve2:, :]
    k3 = k3[:, length-num_ve2:, :]
    v = v[:, length-num_ve2:, :] 

    # Calculate quadruple product attention
    qk1 = tf.einsum('aib,ajb->aij', q, k1)   
    qk2 = tf.einsum('aib,akb->aik', q, k2)
    qk3 = tf.einsum('aib,alb->ail', q, k3)

    k1k2 = tf.einsum('ajb,akb->ajk', k1, k2)
    k1k3 = tf.einsum('ajb,alb->ajl', k1, k3)
    k2k3 = tf.einsum('akb,alb->akl', k2, k3)

    qk1k2k3 = tf.einsum('aij,akl->aijkl', qk1, k2k3)
    qk2k1k3 = tf.einsum('aik,ajl->aikjl', qk2, k1k3)
    qk3k1k2 = tf.einsum('ail,ajk->ailjk', qk3, k1k2)

    pre_logitsvector = qk123 - qk213 + qk312  # <q, k1, k2, k3>
    logits = LayerNorm()(pre_logitsvector) 

    #logits += bias  # No bias since keys only come from virtual entities
    weights_qkkk = multi_softmax(logits, name="weights_qkkk")
    
    if self.train:
      weights_qkkk = tf.nn.dropout(weights_qkkk, 1.0 - self.attention_dropout)

    # computes \sum_{j,k,l}p^{i}_{jkl} C(v[j]\otimes v[k]\otimes v[l])
    Cv = tf.einsum('qrst,ajr->aqstj', self.C_weights, v)
    Cvv = tf.einsum('aqstj,aks->aqtjk', Cv, v)
    Cvvv = tf.einsum('aqtjk,alt->aqjkl', Cvv, v)
    attention_three_output = tf.einsum('aijkl,aqjkl->aiq', weights_qkkk, Cvvv)

    attention_three_output = tf.reshape(attention_three_output, (-1,self.num_heads,length-num_ve1-num_ve2,depth3))
    
    # --> [batch_size, num_heads, length, depth3]
    attention_three_output = tf.concat([attention_three_output,v_virtual_orig],axis=-2) 
    
    # Recombine heads --> [batch_size, length, d3_model]
    attention_three_output = self.combine_heads(attention_three_output)

    # Run the combined outputs through another linear projection layer.
    attention_three_output = self.output_dense_layer(attention_three_output)
    return attention_three_output  # shape (batch_size, length, d3_model)

class SelfAttentionThree(AttentionThree):
  """Multiheaded 3-simplex self-attention layer."""
  def call(self, x, bias, cache=None):
    return super(SelfAttentionThree, self).call(x, x, bias, cache)     