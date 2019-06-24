import tensorflow as tf

from keras import regularizers
from tensorflow.keras.models import Model

# noinspection PyPep8Naming
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Softmax, Embedding, Add, Lambda, Dense, Layer

from position_simple import positional_encoding
from transformer_block import EncoderBlock
from transformer_block import DecoderBlock

# Takes the building blocks from the file transformer_block.py to build a vanilla Transformer.

class Encoder:
    def __init__(self, d1_model, num_heads, transformer_depth, 
                input_vocab_size, drop_rate):
        super(Encoder, self).__init__()

        self.d1_model = d1_model
        self.transformer_depth = transformer_depth

        self.embedding = Embedding(input_vocab_size, d1_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d1_model)

        self.encoder_block = EncoderBlock(d1_model, num_heads, 
                                drop_rate)

        self.enc_layers = [self.encoder_block for i in range(transformer_depth)]

        #self.dropout = layers.Dropout(drop_rate)

    def __call__(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) # (batch_size, input_seq_length, d1_model)
        x *= tf.math.sqrt(tf.cast(self.d1_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        #x = self.dropout(x, training=training)         #no input dropout

        for i in range(self.transformer_depth):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d1_model) 



class Decoder:
    def __init__(self, d1_model, num_heads, transformer_depth, target_vocab_size, 
                 drop_rate):
        super(Decoder, self).__init__()

        self.d1_model = d1_model
        self.transformer_depth = transformer_depth

        self.embedding = Embedding(target_vocab_size, d1_model)
        self.pos_encoding = positional_encoding(target_vocab_size, self.d1_model)

        self.decoder_block = DecoderBlock(d1_model, num_heads,
                               drop_rate)

        self.dec_layers = [self.decoder_block for i in range(transformer_depth)]

        #self.dropout = layers.Dropout(drop_rate)  

    def __call__(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
    
        x = self.embedding(x)  # (batch_size, target_seq_len, d1_model)
        x *= tf.math.sqrt(tf.cast(self.d1_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        #x = self.dropout(x, training=training)         #no input dropout

        for i in range(self.transformer_depth):
            x = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

        #attention_weights['decoder_depth{}_block1'.format(i+1)] = block1
        #attention_weights['decoder_depth{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d1_model)
        return x


class VanillaTransformer(Model):
    """
    A vanilla transformer based on Dehghani et al Universal Transformers (without recurrence)
    """
    def __init__(self, d1_model, num_heads, transformer_depth, 
                    input_vocab_size: int,
                    target_vocab_size: int, 
                    drop_rate: float = 0.1):

        super(VanillaTransformer, self).__init__()
        
        self.encoder = Encoder(d1_model, num_heads, transformer_depth, input_vocab_size,    # dim_ff = Transition Layer dimension
                                 drop_rate)                                                 # which is 4 x d1_model

        self.decoder = Decoder(d1_model, num_heads, transformer_depth, target_vocab_size, 
                                 drop_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def __call__(self, inp, tar, training, enc_padding_mask, 
                     look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d1_model)

        # dec_output.shape == (batch_size, tar_seq_len, d1_model)
        dec_output = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output   

