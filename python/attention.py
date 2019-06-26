import numpy as np
import math

import tensorflow as tf
from keras import backend as K
from keras.engine import Layer
from keras.utils import get_custom_objects

# Based on MultiHeadSelfAttention 
# https://github.com/kpot/keras-transformer/blob/master/keras_transformer/attention.py
# Attention is multi-head by default

class _BaseAttentionOne(Layer):
    """
    Base class for multi-head attention for one-simplices.
    """  
    def __init__(self, num_heads: int, **kwargs):
        """
        :param num_heads: number of attention heads 
        :param kwargs: any extra arguments typical for a Keras layer, such as name, etc.
        """
        #self.mask = mask
        self.num_heads = num_heads
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['num_heads'] = self.num_heads
        #config['mask'] = self.mask
        return config

    def build_output_params(self, d1_model):
        self.output_weights = self.add_weight(
            name='output_weights',
            shape=(d1_model, d1_model),
            initializer='glorot_uniform',
            trainable=True)

    def validate_model_dimensionality(self, d1_model: int):
        if d1_model % self.num_heads != 0:
            raise ValueError(
                f'The size of the last dimension of the input '
                f'({d1_model}) must be evenly divisible by the number'
                f'of the attention heads {self.num_heads}')

    def attention_one(self, pre_q, pre_v, pre_k1, out_seq_len: int, d1_model: int, mask,
                  training=None):
        """
        :param pre_q: (batch_size, q_seq_len, num_heads, d1_model // num_heads)
        :param pre_v: (batch_size, v_seq_len, num_heads, d1_model // num_heads)
        :param pre_k1: (batch_size, k1_seq_len, num_heads, d1_model // num_heads)
        :param mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.
                     The mask has different shapes depending on its type (padding or look ahead). 
        :param out_seq_len: the length of the output sequence
        :param d1_model: dimensionality of the model (by the paper)
        :param training: Passed by Keras. 
        """
        depth1 = d1_model // self.num_heads

        # shaping Q, K1 and V into (batch_size, num_heads, seq_len, depth1)
        q = K.permute_dimensions(pre_q, [0, 2, 1, 3])
        k1 = K.permute_dimensions(pre_k1, [0, 2, 1, 3])
        v = K.permute_dimensions(pre_v, [0, 2, 1, 3])

        q_shape = K.int_shape(q)
        seq_len = q_shape[-2]

        # shaping into (-1, seq_len, depth1) by collapsing batch and head dimensions
        # this allows simultaneous operation on all heads
        q = K.reshape(q, (-1, seq_len, depth1))
        k1 = K.reshape(k1, (-1, seq_len, depth1))
        v = K.reshape(v, (-1, seq_len, depth1))

        pre_logitsvector = tf.einsum('aib,ajb->aij', q, k1)   # <q, k1>      
        norm = K.constant(np.sqrt(depth1), dtype = K.floatx())
        logitsvector = pre_logitsvector / norm

        # Add the mask to the logits vector.  The mask is multiplied with -1e9 (close to negative infinity) because 
        # the mask is summed with the matrix multiplication of Q and K1 and is applied immediately before a softmax.
        if mask is not None:
            logitsvector += (mask * -1e9)

        a = K.softmax(logitsvector, axis=-1)  # (-1, seq_len, seq_len) for computing p^{K,i}_{j}

        # Computes \sum_{j}p^{i}_{j} A(v[j])
        # Av = tf.einsum('qr,ajr->aqj', self.A_weights, v)
        attention_heads = tf.einsum('aij,ajq->aiq', a, v)  # (-1,seq_len,depth1)

        # shaping into (-1, num_heads, seq_len, depth1)
        attention_heads = K.reshape(attention_heads,(-1,self.num_heads,seq_len,depth1)) 

        # shape into (-1, seq_len, self.num_heads, depth1)
        attention_heads = K.permute_dimensions(attention_heads, [0, 2, 1, 3]) 
        
        attention_heads_concatenated = K.reshape(attention_heads,(-1,seq_len,d1_model)) 
        
        return attention_heads_concatenated     # of shape (-1, seq_len, d1_model)
        

    def create_look_ahead_mask(size):
        """
        Makes sure that (when enabled) each position
        (of a decoder's self-attention) cannot attend to subsequent positions.
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

        return mask  # (seq_len, seq_len)

    def create_padding_mask(seq):
        """
        Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat padding as the input. 
        The mask indicates where pad value 0 is present: it outputs a 1 at those locations, and a 0 otherwise.
        """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions so that we can add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)    



class SelfAttentionOne(_BaseAttentionOne):
    """
    Multi-head self-attention for both encoders and decoders.
    Uses only one input.
    """
    def build(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError('Invalid input')

        d1_model = input_shape[-1]

        self.validate_model_dimensionality(d1_model)
        
        # A concatenation of W_q(1), W_q(2),..., W_q(h), W_k1(1), W_k1(2),..., W_k1(h), W_v(1) etc
        # for all h heads.
        self.qkv_weights = self.add_weight(
            name='qkv_weights',
            shape=(d1_model, d1_model * 3),  # * 3 for q, k1 and v
            initializer='glorot_uniform',
            trainable=True)

        self.build_output_params(d1_model)

        return super().build(input_shape)

    def call(self, inputs, mask, **kwargs):
        if not K.is_tensor(inputs):
            raise ValueError(
                'The layer can be called only with one tensor as an argument')
        _, seq_len, d1_model = K.int_shape(inputs)

        # The first thing we need to do is to perform affine transformations
        # of the inputs to get the Queries, Keys and Values.
        qkv = K.dot(inputs, self.qkv_weights)  #(-1, seq_len, d1_model*3)

        qkv = K.reshape(qkv,[-1,d1_model*3])

        # splitting the keys, the values and the queries before further processing
        pre_q, pre_k1, pre_v = [
            K.reshape(
                # K.slice(qkv, (0, i * d1_model), (-1, d1_model)),
                qkv[:, i * d1_model:(i + 1) * d1_model],
                (-1, seq_len, self.num_heads, d1_model // self.num_heads))
            for i in range(3)]

        self_attention_out = self.attention_one(pre_q, pre_v, pre_k1, seq_len, d1_model, mask,
                                       training=kwargs.get('training'))

        return self_attention_out   #(-1, seq_len, d1_model)

    def compute_output_shape(self, input_shape):
        shape_a, seq_len, d1_model = input_shape
        return (shape_a, seq_len, d1_model)




class AttentionOne(_BaseAttentionOne):
    """
    Multi-head attention which can use two inputs:
    First: from the encoder - it's used to project the keys and the values
    Second: from the decoder - used to project the queries.
    """
    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if not (isinstance(input_shape, list) and len(input_shape) == 2):
            raise ValueError(
                'You must call this layer passing a list of two tensors'
                '(for keys/values and queries)')
        values_dim, query_dim = input_shape[0][-1], input_shape[1][-1]
        if query_dim != values_dim:
            raise ValueError(
                f'Both keys/value and query inputs must be '
                f'of the same dimensionality, instead of '
                f'{values_dim} and {query_dim}.')

        d1_model = query_dim

        self.validate_model_dimensionality(d1_model)

        # These weights are concatenated matrices W_k and W_v which
        # are, in turn, concatenated W matrices of keys, and values
        # for each of the heads. So, essentially it's a concatenation of
        # W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        # for all h heads.
        self.kv_weights = self.add_weight(
            name='kv_weights', shape=(d1_model, d1_model * 2),
            initializer='glorot_uniform', trainable=True)
        self.q_weights = self.add_weight(
            name='q_weights', shape=(d1_model, d1_model),
            initializer='glorot_uniform', trainable=True)
        self.build_output_params(d1_model)

        return super().build(input_shape)

    def call(self, inputs, mask, **kwargs):
        if not (isinstance(inputs, list) and len(inputs) == 2):
            raise ValueError(
                'You can call this layer only with a list of two tensors '
                '(for keys/values and queries)')
        key_values_input, query_input = inputs
        _, value_seq_len, d1_model = K.int_shape(key_values_input)
        query_seq_len = K.int_shape(inputs[1])[-2]

        # The first thing we need to do is to perform affine transformations
        # of the inputs to get the Queries, the Keys and the Values.
        kv = K.dot(K.reshape(key_values_input, [-1, d1_model]), self.kv_weights)

        # splitting the keys, the values and the queries before further
        # processing
        pre_k, pre_v = [
            K.reshape(
                # K.slice(kv, (0, i * d1_model), (-1, d1_model)),
                kv[:, i * d1_model: (i + 1) * d1_model],
                (-1, value_seq_len,
                 self.num_heads, d1_model // self.num_heads))
            for i in range(2)]
        pre_q = K.reshape(
            K.dot(K.reshape(query_input, [-1, d1_model]), self.q_weights),
            (-1, query_seq_len, self.num_heads, d1_model // self.num_heads))

        attention_out=self.attention_one(pre_q, pre_v, pre_k, mask, d1_model,
                              training=kwargs.get('training'))

        return attention_out


get_custom_objects().update({
    'SelfAttentionOne': SelfAttentionOne,
    'AttentionOne': AttentionOne,
})
