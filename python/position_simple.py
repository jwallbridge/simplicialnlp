import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

# Based off https://www.tensorflow.org/beta/tutorials/text/transformer

def get_angles(pos, i, d1_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d1_model))

    return pos * angle_rates

def positional_encoding(position, d1_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d1_model)[np.newaxis, :],
                          d1_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)