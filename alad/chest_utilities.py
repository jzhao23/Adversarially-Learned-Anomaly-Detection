"""
CHEST ALAD architecture.
Generator (decoder), encoder and discriminator.
"""
import tensorflow as tf
from utils import sn

learning_rate = 0.0002
batch_size = 32
latent_dim = 100
init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.01)

def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
    else:
        return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

def encoder(x_inp, is_training=False, getter=None, reuse=False,
            do_spectral_norm=True):
    """ Encoder architecture in tensorflow
    Maps the data into the latent space
    Args:
        x_inp (tensor): input data for the encoder.
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not
    Returns:
        net (tensor): last activation layer of the encoder
    """
    layers = sn if do_spectral_norm else tf.layers

    with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):
        x_inp = tf.reshape(x_inp, [-1, 224, 224, 3])

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = layers.conv2d(x_inp,
                           128,
                           kernel_size=4,
                           padding='SAME',
                           strides=2,
                           kernel_initializer=init_kernel,
                           name='conv')
            net = tf.layers.batch_normalization(net,
                                                training=is_training)
            net = leakyReLu(net, name='leaky_relu')

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = layers.conv2d(net,
                           256,
                            kernel_size=4,
                           padding='SAME',
                           strides=2,
                           kernel_initializer=init_kernel,
                           name='conv')
            net = tf.layers.batch_normalization(net,
                                                training=is_training)
            net = leakyReLu(net, name='leaky_relu')

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = layers.conv2d(net,
                           512,
                            kernel_size=4,
                           padding='SAME',
                           strides=2,
                           kernel_initializer=init_kernel,
                           name='conv')
            net = tf.layers.batch_normalization(net,
                                                training=is_training)
            net = leakyReLu(net, name='leaky_relu')

        name_net = 'layer_4' # 28/28
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d(net,
                                   latent_dim,
                                   kernel_size=28,
                                   strides=1,
                                   padding='VALID',
                                   kernel_initializer=init_kernel,
                                   name='conv')
            print("size before squeeze encoder: ", net)
            print("size before squeeze encoder: ", net.get_shape())
            net = tf.squeeze(net, [1, 2])
            print("size after squeeze  encoder: ", net.get_shape())

    return net

def decoder(z_inp, is_training=False, getter=None, reuse=False):
    """ Generator architecture in tensorflow
    Generates data from the latent space
    Args:
        z_inp (tensor): input variable in the latent space
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not
    Returns:
        net (tensor): last activation layer of the generator
    """
    print("z_inp of decoder size: ", z_inp.get_shape())
    with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):
        net = tf.reshape(z_inp, [-1, 1, 1, latent_dim])
        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                     filters=512,
                                     kernel_size=7,
                                     strides=2,
                                     padding='VALID',
                                     kernel_initializer=init_kernel,
                                     name='tconv1')

            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='tconv1/batch_normalization')

            net = tf.nn.relu(net, name='tconv1/relu')

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                     filters=256,
                                     kernel_size=4,
                                     strides=2,
                                     padding='SAME',
                                     kernel_initializer=init_kernel,
                                     name='tconv2')

            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='tconv2/batch_normalization')
            net = tf.nn.relu(net, name='tconv2/relu')

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                     filters=128,
                                     kernel_size=4,
                                     strides=2,
                                     padding='SAME',
                                     kernel_initializer=init_kernel,
                                     name='tconv3')

            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='tconv3/batch_normalization')
            net = tf.nn.relu(net, name='tconv3/relu')
        
        name_net = 'layer_4'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                     filters=128,
                                     kernel_size=4,
                                     strides=2,
                                     padding='SAME',
                                     kernel_initializer=init_kernel,
                                     name='tconv4')

            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='tconv4/batch_normalization')
            net = tf.nn.relu(net, name='tconv4/relu')

        name_net = 'layer_5'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                     filters=128,
                                     kernel_size=4,
                                     strides=2,
                                     padding='SAME',
                                     kernel_initializer=init_kernel,
                                     name='tconv5')

            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='tconv5/batch_normalization')
            net = tf.nn.relu(net, name='tconv5/relu')

        name_net = 'layer_6'
        with tf.variable_scope(name_net):
            net = tf.layers.conv2d_transpose(net,
                                     filters=3,
                                     kernel_size=4,
                                     strides=2,
                                     padding='SAME',
                                     kernel_initializer=init_kernel,
                                     name='tconv6')

            print("size decoder: ", net)
            print("size decoder: ", net.get_shape())
            net = tf.tanh(net, name='tconv6/tanh')

        

    return net


def discriminator_xz(x_inp, z_inp, is_training=False, getter=None, reuse=False,
                     do_spectral_norm=True):
    """ Discriminator architecture in tensorflow
    Discriminates between pairs (E(x), x) and (z, G(z))
    Args:
        x_inp (tensor): input data for the discriminator.
        z_inp (tensor): input variable in the latent space
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not
    Returns:
        logits (tensor): last activation layer of the discriminator (shape 1)
        intermediate_layer (tensor): intermediate layer for feature matching
    """
    layers = sn if do_spectral_norm else tf.layers
    print("x_inp size: ", x_inp.get_shape())
    print("z_inp size: ", z_inp.get_shape())

    with tf.variable_scope('discriminator_xz', reuse=reuse, custom_getter=getter):
        name_net = 'x_layer_1'
        with tf.variable_scope(name_net):
            x = layers.conv2d(x_inp,
                                 filters=128,
                                 kernel_size=4,
                                 strides=2,
                                 padding='SAME',
                                 kernel_initializer=init_kernel,
                                 name='conv1')

            x = leakyReLu(x, 0.2, name='conv1/leaky_relu')

        name_net = 'x_layer_2'
        with tf.variable_scope(name_net):
            x = layers.conv2d(x,
                                 filters=256,
                                 kernel_size=4,
                                 strides=2,
                                 padding='SAME',
                                 kernel_initializer=init_kernel,
                                 name='conv2')

            x = tf.layers.batch_normalization(x,
                                              training=is_training,
                                              name='conv2/batch_normalization')

            x = leakyReLu(x, 0.2, name='conv2/leaky_relu')

        name_net = 'x_layer_3'
        with tf.variable_scope(name_net):
            x = layers.conv2d(x,
                                 filters=512,
                                 kernel_size=4,
                                 strides=2,
                                 padding='SAME',
                                 kernel_initializer=init_kernel,
                                 name='conv3')

            x = tf.layers.batch_normalization(x,
                                              training=is_training,
                                              name='conv3/batch_normalization')

            x = leakyReLu(x, 0.2, name='conv3/leaky_relu')
        print("x before reshape size: ", x.get_shape())
        x = tf.reshape(x, [-1,1,1,512*4*4])
        print("x after reshape size: ", x.get_shape())

        z = tf.reshape(z_inp, [-1, 1, 1, latent_dim])
        print("z size: ", z.get_shape())
        name_net = 'z_layer_1'
        with tf.variable_scope(name_net):
            z = layers.conv2d(z,
                                 filters=512,
                                 kernel_size=1,
                                 strides=1,
                                 padding='SAME',
                                 kernel_initializer=init_kernel,
                                 name='conv')
            z = leakyReLu(z)
            z = tf.layers.dropout(z, rate=0.2, training=is_training,
                                  name='dropout')

        name_net = 'z_layer_2'
        with tf.variable_scope(name_net):
            z = layers.conv2d(z,
                                 filters=512,
                                 kernel_size=1,
                                 strides=1,
                                 padding='SAME',
                                 kernel_initializer=init_kernel,
                                 name='conv')
            z = leakyReLu(z)
            z = tf.layers.dropout(z, rate=0.2, training=is_training,
                                  name='dropout')
        print("z after layers size: ", z.get_shape())

        y = tf.concat([x, z], axis=-1)
        print("y before size: ", y.get_shape()) #1,1,8704 (512+8192)
        name_net = 'y_layer_1'
        with tf.variable_scope(name_net):
            y = layers.conv2d(y,
                                 filters=1024,
                                 kernel_size=1,
                                 strides=1,
                                 padding='SAME',
                                 kernel_initializer=init_kernel,
                                 name='conv')
            y = leakyReLu(y)
            y = tf.layers.dropout(y, rate=0.2, training=is_training,
                                  name='dropout')

        intermediate_layer = y
        print("intermediate layer size: ", intermediate_layer.get_shape()) #1,1,1024

        name_net = 'y_layer_2'
        with tf.variable_scope(name_net):
            y = tf.layers.conv2d(y,
                                 filters=1,
                                 kernel_size=1,
                                 strides=1,
                                 padding='SAME',
                                 kernel_initializer=init_kernel,
                                 name='conv')
        print("y before final squeeze size: ", y.get_shape()) #1,1,1
        logits = tf.squeeze(y)

    return logits, intermediate_layer

def discriminator_xx(x, rec_x, is_training=False, getter=None, reuse=False,
                     do_spectral_norm=True):
    """ Discriminator architecture in tensorflow
    Discriminates between (x,x) and (x,rec_x)
    Args:
        x (tensor): input from the data space
        rec_x (tensor): reconstructed data
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not
    Returns:
        logits (tensor): last activation layer of the discriminator
        intermediate_layer (tensor): intermediate layer for feature matching
    """
    layers = sn if do_spectral_norm else tf.layers

    with tf.variable_scope('discriminator_xx', reuse=reuse, custom_getter=getter):
        print("x: ", x.get_shape())
        print("rec_x: ", rec_x.get_shape())
        net = tf.concat([x, rec_x], axis=1)

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = layers.conv2d(net,
                           filters=64,
                           kernel_size=5,
                           strides=2,
                           padding='SAME',
                           kernel_initializer=init_kernel,
                           name='conv1')

            net = leakyReLu(net, 0.2, name='conv1/leaky_relu')

            net = tf.layers.dropout(net, rate=0.2, training=is_training,
                                  name='dropout')
        with tf.variable_scope(name_net, reuse=True):
            weights = tf.get_variable('conv1/kernel')

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = layers.conv2d(net,
                           filters=128,
                           kernel_size=5,
                           strides=2,
                           padding='SAME',
                           kernel_initializer=init_kernel,
                           name='conv2')
            net = leakyReLu(net, 0.2, name='conv2/leaky_relu')

            net = tf.layers.dropout(net, rate=0.2, training=is_training,
                                  name='dropout')

        net = tf.contrib.layers.flatten(net)

        intermediate_layer = net
        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=1,
                                   kernel_initializer=init_kernel,
                                   name='fc')

            logits = tf.squeeze(net)

    return logits, intermediate_layer

def discriminator_zz(z, rec_z, is_training=False, getter=None, reuse=False,
                     do_spectral_norm=True):
    """ Discriminator architecture in tensorflow
    Discriminates between (z,z) and (z,rec_z)
    Args:
        z (tensor): input from the latent space
        rec_z (tensor): reconstructed data
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not
    Returns:
        logits (tensor): last activation layer of the discriminator
        intermediate_layer (tensor): intermediate layer for feature matching
    """
    layers = sn if do_spectral_norm else tf.layers

    with tf.variable_scope('discriminator_zz', reuse=reuse,
                           custom_getter=getter):

        y = tf.concat([z, rec_z], axis=-1)

        name_net = 'y_layer_1'
        with tf.variable_scope(name_net):
            y = layers.dense(y, units=64, kernel_initializer=init_kernel,
                                 name='fc')

            y = leakyReLu(y)
            y = tf.layers.dropout(y, rate=0.2, training=is_training,
                                  name='dropout')

        name_net = 'y_layer_2'
        with tf.variable_scope(name_net):
            y = layers.dense(y,
                                 units=32,
                                 kernel_initializer=init_kernel,
                                 name='fc')

            y = leakyReLu(y)
            y = tf.layers.dropout(y, rate=0.2, training=is_training,
                                  name='dropout')

        intermediate_layer = y

        name_net = 'y_layer_3'
        with tf.variable_scope(name_net):
            y = tf.layers.dense(y,
                                 units=1,
                                 kernel_initializer=init_kernel,
                                 name='fc')

            logits = tf.squeeze(y)

    return logits, intermediate_layer