"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    """
        Composite function, consists of 3 consecutive operations:
            1. Batch normalization
            2. ReLU
            3. 3 x 3 convolution

    :param current:
    :param num_outputs:
    :param kernel_size:
    :param scope:
    :return:
    """
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    """
        Although each layer only produces k output feature-maps, it typically has many more inputs.
        It has been noted that a 1 x 1 convolution can be introduced as bottleneck layer
            before each 3 x 3 convolution to reduce the number of input feature-maps, and thus to improve computational efficiency.
    :param net:
    :param layers: 层数。每个layer包含一个稠密链接的组合和一个转换层（transition layer）。
    :param growth: growth可以控制每层输出的特征数，防止层数增加的同时产生过多的特征，从而影响模型性能。
    :param scope:
    :return:
    """
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
        net = add_transition_layer(net, scope=scope + "_tran_" + str(idx))
    return net


def add_transition_layer(net, scope):
    """
        The transition layers used in our experiments consist of:
            a batch normalization layer
            and an 1 x 1 convolutional layer
            followed by a 2 x 2 average pooling layer.

        Compression:
            To further improve model compactness, we can reduce the number of feature-maps at transition layers.
            We refer the DenseNEt with theta < 1 as DenseNet-C, and we set theta = 0.5 in our experiment.
            When both the bottleneck and transition layers with theta < 1 are used, we refer to our model as DenseNet-BC.
    :param net:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        net = slim.batch_norm(net, scope=scope + "_bn")
        net = slim.conv2d(net, 12, [1, 1], scope=scope + "_conv")
        net = slim.avg_pool2d(net, [2, 2])
        net = slim.dropout(net, keep_prob=0.5, scope=scope + "_dropout")
        return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 12
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            pass
            ##########################
            with tf.variable_scope("Layer_0"):
                """
                第一层对输入数据进行处理：
                    Before entering the first dense block, a convolution with 16(or twice the growth rate for DenseNet-BC)
                    output channels is performed on the input images.
                """
                conv0 = slim.conv2d(images, 16, [7, 7])
                pool0 = slim.avg_pool2d(conv0, 2)
                net = block(pool0, 3, 12)

            # end_point = 'Block_1'
            # with tf.variable_scope(end_point):
            #     net1_input = pool0
            #     net1 = block(net1_input, 12, growth, scope='Net_1')
            #     net1 = add_transition_layer(net1, scope='Transition_1')
            #
            # # end_points[end_point] = net1
            #
            # end_point = 'Block_2'
            # with tf.variable_scope(end_point):
            #     net2_input = net1
            #     net2 = block(net2_input, 12, growth, scope='Net_2')
            #     net2 = add_transition_layer(net2, scope='Transition_2')
            # # end_points[end_point] = net2
            #
            # end_point = 'Block_3'
            # with tf.variable_scope(end_point):
            #     net3_input = net2
            #     net3 = block(net3_input, 12, growth, scope='Net_3')
            #     net3 = add_transition_layer(net3, scope='Transition_3')
            # # end_points[end_point] = net3

            # 接入输出层
            end_point = 'Output'
            with tf.variable_scope(end_point):
                output = slim.batch_norm(net, is_training=True)
                output = tf.nn.relu(output)
                output_kernel = int(output.get_shape()[-2])
                output = slim.avg_pool2d(output, [output_kernel, output_kernel])
                output = slim.fully_connected(output, num_classes)

                logits = tf.reshape(output, [-1, num_classes])
            # end_points[end_point] = logits

            ##########################

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
