import tensorflow as tf


def init_weight(shape):
    """
    Init weight parameter.
    """
    w = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
    return tf.Variable(w)


def init_bias(shape):
    """
    Init bias parameter.
    """
    b = tf.zeros(shape)
    return tf.Variable(b)


if __name__ == "__main__":
    input = tf.Variable(tf.constant(1.0, shape=[1, 32, 32, 1]))
    conv1_w = init_weight([5, 5, 1, 6])
    conv1_b = init_bias(6)
    conv1 = tf.nn.conv2d(input, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    pool1 = tf.nn.avg_pool2d(conv1, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    pool1 = tf.nn.relu(pool1)
    conv2_w = init_weight([5, 5, 6, 16])
    conv2_b = init_bias(16)
    conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    pool2 = tf.nn.avg_pool2d(conv2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    pool2 = tf.nn.relu(pool2)
    fc1_w = init_weight([5*5*16, 120])
    fc1_b = init_bias(120)
    fc1 = tf.add(tf.matmul(tf.reshape(pool2, [-1, 5*5*16]), fc1_w), fc1_b)
    fc2_w = init_weight([120, 84])
    fc2_b = init_bias(84)
    fc2 = tf.add(tf.matmul(fc1, fc2_w), fc2_b)
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.tile(fc2, [10, 1])
    out_w = init_weight([10, 84])
    out = tf.reduce_sum(tf.square(tf.subtract(fc2, out_w)), 1)

    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        print(sess.run(out).shape)
