import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x,
          dropout_prob = 1.0,
          image_depth = 3,
          conv1_output_depth = 12, # 6,
          conv2_output_depth = 20, # 16,
          fc1_output_shape = 160, # 128,
          fc2_output_shape = 100 # 84
         ):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # (height, width, input_depth, output_depth)
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, image_depth, conv1_output_depth), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(conv1_output_depth))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, conv1_output_depth, conv2_output_depth), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(conv2_output_depth))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_input_shape = 5 * 5 * conv2_output_depth
    fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_input_shape, fc1_output_shape), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(fc1_output_shape))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # Dropout
    fc1 = tf.nn.dropout(fc1, dropout_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(fc1_output_shape, fc2_output_shape), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(fc2_output_shape))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # Dropout
    fc2 = tf.nn.dropout(fc2, dropout_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(fc2_output_shape, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    regularizer = tf.nn.l2_loss(fc1_W) + \
      tf.nn.l2_loss(fc1_b) + \
      tf.nn.l2_loss(fc2_W) + \
      tf.nn.l2_loss(fc2_b) + \
      tf.nn.l2_loss(fc3_W) + \
      tf.nn.l2_loss(fc3_b)

    return logits, regularizer

