import tensorflow as tf
from tensorflow.contrib import layers


def encoder(input_tensor, output_size):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        A tensor that expresses the encoder network
    '''
    net1 = tf.reshape(input_tensor, [-1, 32, 32, 1])
    net3 = layers.conv2d(net1, 32, 5, stride=2)
    net4 = layers.conv2d(net3, 32, 5, stride=2)
    net5 = layers.conv2d(net4, 64, 5, stride=2)
    # net6 = layers.conv2d(net5, 128, 5, stride=2, padding='VALID')
    net7 = layers.dropout(net5, keep_prob=0.9)
    net = layers.flatten(net7)
    return layers.fully_connected(net, output_size, activation_fn=None)


def discriminator(input_tensor):
    '''Create a network that discriminates between images from a dataset and
    generated ones.

    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    '''

    return encoder(input_tensor, 1)


def decoder(input_tensor):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        A tensor that expresses the decoder network
    '''

    net1 = tf.expand_dims(input_tensor, 1)
    net2 = tf.expand_dims(net1, 1)
    net3 = layers.conv2d_transpose(net2, 128, 3, padding='VALID')
    net4 = layers.conv2d_transpose(net3, 64, 5, padding='VALID')
    net5 = layers.conv2d_transpose(net4, 32, 5, stride=2)
    net5 = layers.conv2d_transpose(net5, 32, 5, stride=2)
    net5 = layers.conv2d_transpose(net5, 32, 5, padding='VALID')
    net6 = layers.conv2d_transpose(
        net5, 1, 5, stride=1, activation_fn=tf.nn.sigmoid)
    net = layers.flatten(net6)
    return net
