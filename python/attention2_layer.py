"""Implementation of multiheaded attention and self-attention layers.
Based on 
https://github.com/tensorflow/models/blob/master/official/transformer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
#from keras import initializers
#from keras.layers import Layer, activations
from tensorflow.keras import backend as K

def multi_softmax(target, axis, name=None):
  with tf.name_scope(name, 'softmax', values=[target]):
    max_axis = tf.reduce_max(target, axis, keep_dims=True)
    target_exp = tf.exp(target-max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
    softmax = target_exp / normalize
    return softmax

class AttentionTwo(tf.layers.Layer):  
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, d2_model: int, num_heads, num_vir_entities: int, attention_dropout, train):
    if d2_model % num_heads != 0:
      raise ValueError("d2_model must be evenly divisible by the number of "
                       "heads.")

    super(AttentionTwo, self).__init__()
    self.hidden_size = hidden_size
    self.d2_model = d2_model
    self.num_heads = num_heads
    self.num_vir_entities = num_vir_entities
    self.attention_dropout = attention_dropout
    self.train = train

    self.output_dense_layer = tf.layers.Dense(d2_model, use_bias=False, 
                                              name="output_transform2")

  def build(self, input_shape):
    self.q_weights = tf.get_variable("q", 
           shape=[self.hidden_size, self.d2_model], initializer=None, trainable=True)
    self.k1_weights = tf.get_variable("k1", 
           shape=[self.hidden_size, self.d2_model], initializer=None, trainable=True)
    self.k2_weights = tf.get_variable("k2", 
           shape=[self.hidden_size, self.d2_model], initializer=None, trainable=True)                                       
    self.v_weights = tf.get_variable("v", 
           shape=[self.hidden_size, self.d2_model], initializer=None, trainable=True)
        
    self.B_weights = tf.get_variable("B",
            shape=[self.d2_model // self.num_heads, self.d2_model // self.num_heads,
                                    self.d2_model // self.num_heads], trainable=True)                         

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, d2_model]
    Returns:
      A tensor with shape [batch_size, num_heads, length, d2_model/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth2 = (self.d2_model // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth2])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, d2_model/num_heads]
    Returns:
      A tensor with shape [batch_size, length, d2_model]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth2]
      return tf.reshape(x, [batch_size, length, self.d2_model])

  def call(self, x, y, bias, cache=None):
    """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x + num_ve, hidden_size]
      y: a tensor with shape [batch_size, length_y + num_ve, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results    CHECK !!!!
              of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      AttentionTwo layer output with shape [batch_size, length_x + num_ve, d2_model]
    """
    num_ve = self.num_vir_entities
    length = tf.shape(x)[1]  # input_length + num_ve
    depth2 = (self.d2_model // self.num_heads)
    
    # Linearly project the query (q), key one (k1), key two (k2) and value (v) 
    # using different learned projections. 
    q = K.dot(x, self.q_weights) # shape (batch_size, length, d2_model)
    k1 = K.dot(y, self.k1_weights)
    k2 = K.dot(y, self.k2_weights)
    v = K.dot(y, self.v_weights)
    
    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k1 = tf.concat([cache["k1"], k1], axis=1)
      k2 = tf.concat([cache["k2"], k1], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k1"] = k1
      cache["k2"] = k2
      cache["v"] = v

    # Split q, k1, k2, v into heads.
    q = self.split_heads(q) # shape (batch_size, num_heads, length, depth2)
    k1 = self.split_heads(k1)
    k2 = self.split_heads(k2)
    v = self.split_heads(v)

    v_virtual_orig = v[:,:,length-num_ve:,:] # shape (batch_size, num_heads, length-num_ve, depth2)
    
    # collapse the batch dimension and head dimensions to operate simultaneously on all heads. 
    q = tf.reshape(q, (-1, length, depth2)) # shape (-1, length, depth2)
    k1 = tf.reshape(k1, (-1, length, depth2))
    k2 = tf.reshape(k2, (-1, length, depth2))
    v = tf.reshape(v, (-1, length, depth2))
    
    # We generate queries only for standard entities, and keys and
    # values only for virtual entities.
    q = q[:, :length-num_ve, :] # shape (batch_size, len-num_ve, depth2)
    k1 = k1[:, length-num_ve:, :] 
    k2 = k2[:, length-num_ve:, :]
    v = v[:, length-num_ve:, :] 
   
    # Calculate triple product attention
    qk1 = tf.einsum('aib,ajb->aij', q, k1) # shape (batch_size, len-num_ve, num_ve)
    qk2 = tf.einsum('aib,akb->aik', q, k2)
    k1k2 = tf.einsum('ajb,akb->ajk', k1, k2)
    
    qq = tf.einsum('aib,aib->ai', q, q)
    k1k1 = tf.einsum('ajb,ajb->aj', k1, k1)
    k2k2 = tf.einsum('akc,akc->ak', k2, k2)

    qk1k2k2 = tf.einsum('aij,ak->aijk', tf.square(qk1), k2k2) # shape (batch_size, len-num_ve, num_ve, num_ve)
    k1k2qq = tf.einsum('ajk,ai->aijk', tf.square(k1k2), qq)
    qk2k1k1 = tf.einsum('aik,aj->aijk', tf.square(qk2), k1k1)

    qk1_e = tf.expand_dims(qk1, axis=3)  # qk1_e = tf.einsum('aij->aijk',qk1)
    qk2_e = tf.expand_dims(qk2, axis=2)  # qk2_e = tf.einsum('aik->aijk',qk2)
    k1k2_e = tf.expand_dims(k1k2, axis=1)  # k1k2_e = tf.einsum('ajk->aijk',k1k2)
    
    pre_logitsvector = qk1k2k2 + k1k2qq + qk2k1k1 - 2 * qk1_e * qk2_e * k1k2_e # shape (batch_size, len-num_ve, num_ve, num_ve) 
    
    #norm = tf.sqrt(tf.constant(6*depth2**2 + 4*depth2**3, dtype=tf.float32))
    #pre_logitsvector = tf.sqrt(pre_logitsvector)
    #logits = pre_logitsvector / norm
    logits = LayerNorm()(pre_logitsvector)

    #logits += bias # No bias since keys only come from virtual entities 
    weights_qkk = multi_softmax(logits, axis=[-2,-1], name="weights_qkk")
    
    if self.train:
      weights_qkk = tf.nn.dropout(weights_qkk, 1.0 - self.attention_dropout)
      B_weights = tf.nn.dropout(self.B_weights, 1.0 - self.attention_dropout)
    
    Bv = tf.einsum('qrs,ajr->aqsj',self.B_weights,v)
    Bvv = tf.einsum('aqsj,aks->aqjk',Bv,v)
    attention_two_output = tf.einsum('aijk,aqjk->aiq', weights_qkk, Bvv) # shape (batch_size, length-num_ve, depth2)
    
    attention_two_output = tf.reshape(attention_two_output, (-1,self.num_heads,length-num_ve,depth2))
    
    # --> [batch_size, num_heads, length, depth2]
    attention_two_output = tf.concat([attention_two_output,v_virtual_orig],axis=-2) 
    
    # Recombine heads --> [batch_size, length, d2_model]
    attention_two_output = self.combine_heads(attention_two_output)

    # Run the combined outputs through another linear projection layer.
    attention_two_output = self.output_dense_layer(attention_two_output)
    return attention_two_output  # shape (batch_size, length, d2_model)

class SelfAttentionTwo(AttentionTwo):
  """Multiheaded self-attention layer."""
  def call(self, x, bias, cache=None):
    return super(SelfAttentionTwo, self).call(x, x, bias, cache)         
    
