import math
from typing import Union, Callable, Optional

from keras.layers import Layer, Add, activations, Dropout
from keras import initializers

# noinspection PyPep8Naming
from tensorflow.keras import backend as K
from keras.utils import get_custom_objects

from attention import _BaseAttentionOne
from attention import SelfAttentionOne
from attention import AttentionOne

# Based on Keras transformer
# https://github.com/kpot/keras-transformer/blob/master/keras_transformer/transformer.py

def gelu(x):
    """
    GELU activation, described in paper "Gaussian Error Linear Units (GELUs)"
    https://arxiv.org/pdf/1606.08415.pdf
    """
    c = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + K.tanh(c * (x + 0.044715 * K.pow(x, 3))))


class LayerNormalization(Layer):
    """
    Implementation of Layer Normalization (https://arxiv.org/abs/1607.06450).
    """
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(
            K.square(inputs - mean), axis=self.axis, keepdims=True)
        epsilon = K.constant(1e-5, dtype=K.floatx())
        normalized_inputs = (inputs - mean) / K.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class TransformerTransition(Layer):
    """
    Transformer transition function. The same function is used both
    in classical in Universal Transformers. Except that in Universal
    Transformer it is also shared between time steps.
    """

    def __init__(self, activation: Union[str, Callable],
                 size_multiplier: int = 4, **kwargs):
        """
        :param activation: activation function. Must be a string or a callable.
        :param size_multiplier: How big the hidden dimension should be.
          Most of the implementation use transition functions having 4 times
          more hidden units than the model itself. eg. Vanilla Transformer : 4 x 512 = 2048
        :param kwargs: Keras-specific layer arguments.
        """
        self.activation = activations.get(activation)
        self.size_multiplier = size_multiplier
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['activation'] = activations.serialize(self.activation)
        config['size_multiplier'] = self.size_multiplier
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        d1_model = input_shape[-1]
        self.weights1 = self.add_weight(
            name='weights1',
            shape=(d1_model, self.size_multiplier * d1_model),
            initializer='glorot_uniform',
            trainable=True)
        self.biases1 = self.add_weight(
            name='biases1',
            shape=(self.size_multiplier * d1_model,),
            initializer='zeros',
            trainable=True)
        self.weights2 = self.add_weight(
            name='weights2',
            shape=(self.size_multiplier * d1_model, d1_model),
            initializer='glorot_uniform',
            trainable=True)
        self.biases2 = self.add_weight(
            name='biases2',
            shape=(d1_model,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        d1_model = input_shape[-1]
        step1 = self.activation(
            K.bias_add(
                K.dot(K.reshape(inputs, (-1, d1_model)),
                      self.weights1),
                self.biases1,
                data_format='channels_last'))
        step2 = K.bias_add(
            K.dot(step1, self.weights2),
            self.biases2,
            data_format='channels_last')
        result = K.reshape(step2, (-1,) + input_shape[-2:])
        return result


class EncoderBlock:  # NOTE : Universal transformer paper actually places Residual connection AFTER Dropout
    """
    A pseudo-layer following description from the "Universal Transformers" paper.
    Each encoder block is, essentially:
    
    - Multi-head self-attention (unmasked, with attention dropout, but w/o input dropout)
    - Residual connection
    - Dropout
    - Layer normalization
    
    - Transition function
    - Residual connection
    - Dropout
    - Layer normalization
    
    IMPORTANT: The older Transformer 2017 model ("Attention is all you need")
    uses slightly different order of operations. A quote from the paper:
        "We apply dropout [33] to the output of each sub-layer,
         before it is added to the sub-layer input and normalized"
    while the Universal Transformer paper puts dropout one step *after*
    the sub-layers's output was added to its input (Figure 4 in the paper).
    In this code the order from the Universal Transformer is used, as arguably
    more reasonable. You can use classical Transformer's (2017) way of
    connecting the pieces by passing vanilla_wiring=True to the constructor.
    """
    def __init__(self, name: str, num_heads: int, 
                 drop_rate: float = 0, 
                 activation: Optional[Union[str, Callable]] = 'gelu'):

        self.mha = SelfAttentionOne(num_heads, 
                                       name=f'{name}_self_attention_one')

        self.norm1_layer = LayerNormalization(name=f'{name}_normalization1')

        self.dropout_layer = (
            Dropout(drop_rate, name=f'{name}_dropout')
            if drop_rate > 0
            else lambda x: x)

        self.norm2_layer = LayerNormalization(name=f'{name}_normalization2')

        self.transition_layer = TransformerTransition(
            name=f'{name}_transition', activation=activation)

        self.addition_layer = Add(name=f'{name}_add')

    def __call__(self, _input, mask):
        output = self.mha(_input, mask)
        post_residual1 = (
            self.dropout_layer
               (self.addition_layer([_input, output])))
        norm1_output = self.norm1_layer(post_residual1)
        
        output = self.transition_layer(norm1_output)
        post_residual2 = (
              self.dropout_layer(
                  self.addition_layer([norm1_output, output])))
        output = self.norm2_layer(post_residual2)

        return output


class DecoderBlock:
    """
    A pseudo-layer combining together all nuts and bolts to assemble
    a complete section of both the Transformer and the Universal Transformer
    models, following description from the "Universal Transformers" paper.
    Each decoder block is, essentially:
    
    - Multi-head self-attention (masked, with attention dropout, but w/o input dropout)
    - Residual connection
    - Dropout
    - Layer normalization
    
    - Multi-head attention
    - Residual connection
    - Dropout
    - Layer normalization
    
    - Transition function
    - Residual connection
    - Dropout
    - Layer normalization
    """

    def __init__(self, name: str, num_heads: int,
                 drop_rate: float = 0,
                 activation: Optional[Union[str, Callable]] = 'gelu'):

        self.mha1 = SelfAttentionOne(num_heads,
                           name=f'{name}_self_attention1_one')

        self.mha2 = AttentionOne(num_heads, 
                           name=f'{name}_self_attention2_one')

        self.norm1_layer = LayerNormalization(name=f'{name}_normalization1')

        self.dropout_layer = (
            Dropout(drop_rate, name=f'{name}_dropout')
            if drop_rate > 0
            else lambda x: x)

        self.norm2_layer = LayerNormalization(name=f'{name}_normalization2')

        self.norm3_layer = LayerNormalization(name=f'{name}_normalization3')

        self.transition_layer = TransformerTransition(
            name=f'{name}_transition', activation=activation)

        self.addition_layer = Add(name=f'{name}_add')


    def __call__(self, _input, look_ahead_mask, padding_mask):

        output = self.mha1(_input, look_ahead_mask)
        post_residual1 = (
            self.dropout_layer(self.addition_layer([_input, output])))
        norm1_output = self.norm1_layer(post_residual1)

        # Takes two tensors : kv from EncoderBlock output + q from norm1_output
        output = self.mha2(enc_output, norm1_output, padding_mask)   
        post_residual2 = (
            self.dropout_layer(self.addition_layer([inputs, output])))
        norm2_output = self.norm1_layer(post_residual2)

        output = self.transition_layer(norm2_output)
        post_residual3 = (
            self.dropout_layer(
                self.addition_layer([norm2_output, output])))
        output = self.norm3_layer(post_residual3)

        return output



get_custom_objects().update({
    'LayerNormalization': LayerNormalization,
    'TransformerTransition': TransformerTransition,
    'gelu': gelu,
})