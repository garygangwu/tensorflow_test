import sys
import time
import math
from sklearn.utils import shuffle
import tensorflow as tf
from traffic_sign_data_loader import *
from traffic_sign_training_utils import *
import matplotlib.pyplot as plt

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DROPOUT_PROB = 0.5
BETA = 0.01
MODEL_SAVE_FOLD = './tf_models/'

def print_out_training_data_summary(X_train, y_train, X_valid, y_valid, X_test, y_test):
  n_train = len(X_train)
  n_valid = len(X_valid)
  n_test = len(X_test)
  image_shape = X_train[0].shape
  n_classes = max(y_train) - min(y_train) + 1

  print("Number of training examples =", n_train)
  print("  shape {}...".format(X_train.shape))
  print("  shape {}...".format(y_train.shape))
  print("Number of validation examples =", n_valid)
  print("  shape {}...".format(X_valid.shape))
  print("  shape {}...".format(y_valid.shape))
  print("Number of testing examples =", n_test)
  print("  shape {}...".format(X_test.shape))
  print("  shape {}...".format(y_test.shape))
  print("Image data shape =", image_shape)
  print("Number of classes =", n_classes)


def train(X_train, y_train, X_valid, y_valid, X_test, y_test,
          x, y, prob, training_operation, loss_operation, accuracy_operation, saver):
  session = tf.Session()
  session.run(tf.global_variables_initializer())
  num_examples = len(X_train)

  print("Training...")
  print('')

  model_infos = []
  for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, num_examples, BATCH_SIZE):
      end = offset + BATCH_SIZE
      batch_x, batch_y = X_train[offset:end], y_train[offset:end]
      _, loss = session.run([training_operation, loss_operation],
                            feed_dict = {x: batch_x, y: batch_y, prob: DROPOUT_PROB})

    print("EPOCH {} ...".format(i))
    print("Loss {:.3f}".format(loss))
    validation_accuracy = evaluate(X_valid, y_valid, x, y, prob, accuracy_operation, session)
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    train_accuracy = evaluate(X_train, y_train, x, y, prob, accuracy_operation, session)
    print("Train Accuracy = {:.3f}".format(train_accuracy))
    print('')
    saver.save(session, MODEL_SAVE_FOLD + 'lenet', global_step=i)
    model_infos.append({'step': i, 'loss': loss, 'validation_accuracy': validation_accuracy})

  sorted_model_infos = sorted(model_infos,
                              key=lambda info:info['validation_accuracy'],
                              reverse=True)
  selected_model = sorted_model_infos[0]
  selected_model_file = MODEL_SAVE_FOLD + 'lenet-' + str(selected_model['step'])

  saver.restore(session, selected_model_file)
  saver.save(session, MODEL_SAVE_FOLD + 'final')
  print("Final Model saved : " + selected_model_file)


def evaluate(X_data, y_data, x, y, prob, accuracy_operation, session):
  num_examples = len(X_data)
  total_accuracy = 0
  for offset in range(0, num_examples, BATCH_SIZE):
    batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
    accuracy = session.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, prob: 1.0})
    total_accuracy += (accuracy * len(batch_x))
  return total_accuracy / num_examples


def check_test_accuracy(X_test, y_test, x, y, prob,
                        accuracy_operation, logits, saver):
  #import scipy.misc

  session = tf.Session()
  model_file = tf.train.latest_checkpoint(MODEL_SAVE_FOLD)
  print('Load model file from ' + model_file)
  saver.restore(session, model_file)

  test_accuracy, logits_results = session.run(
    [accuracy_operation, logits],
    feed_dict={x: X_test, y: y_test, prob: 1.0})

  print('')
  print("Test Accuracy = {:.3f}".format(test_accuracy))

  predictions = session.run(tf.argmax(logits_results, 1))
  for i in range(len(predictions)):
     if predictions[i] != y_test[i]:
       print('Prediction of label {}, correct label is {}'.format(predictions[i], y_test[i]))
  #     filename = 'images/bad_pred_' + str(y_test[i]) + '_' + str(predictions[i]) + '.png'
  #     scipy.misc.imsave(filename, X_test[i])


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(labels, featuremaps, file_name_prefix,
                     activation_min=-1, activation_max=-1 ,plt_num=1):

  num_layer = math.ceil(featuremaps.shape[3] * 1.0 / 10)
  for k in xrange(len(labels)):
    label = labels[k]
    plt.figure(plt_num, figsize=(22,4.5))
    #plt.clf()
    for i in xrange(featuremaps.shape[3]):
      plt.subplot(num_layer, 10, i+1) # sets the number of feature maps to show on each row and column
      plt.title('FeatureMap ' + str(i)) # displays the feature map number
      if activation_min != -1 & activation_max != -1:
        plt.imshow(featuremaps[k,:,:,i], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
      elif activation_max != -1:
        plt.imshow(featuremaps[k,:,:,i], interpolation="nearest", vmax=activation_max, cmap="gray")
      elif activation_min !=-1:
        plt.imshow(featuremaps[k,:,:,i], interpolation="nearest", vmin=activation_min, cmap="gray")
      else:
        plt.imshow(featuremaps[k,:,:,i], interpolation="nearest", cmap="gray")

    filename = file_name_prefix + '_' + str(label) + '.png'
    plt.savefig(filename, bbox_inches='tight')
    print("Saved to " + filename)

def convert_png_to_ppm(img):
  new_img = np.zeros((32,32,3))
  for i in xrange(img.shape[0]):
    for j in xrange(img.shape[1]):
      new_img[i][j] = 255 * img[i][j][:3]
  return normalize_images(new_img)

def test_real_images(x, y, prob, accuracy_operation, logits, conv1_x, conv2_x, saver):
  import matplotlib.image as mpimg
  import os

  test_image_dir = './test_images/'
  input_files = os.listdir(test_image_dir)
  X_test = []
  y_test = []
  for filename in input_files:
    if not '.png' in filename or not filename[0].isdigit():
      continue
    img = convert_png_to_ppm(mpimg.imread(test_image_dir + filename))
    if img.shape != (32, 32, 3):
      continue
    label = int(filename.split('.')[0])
    X_test.append(img)
    y_test.append(label)
  X_test, y_test = shuffle(X_test, y_test)
  X_test, y_test = X_test[:5], y_test[:5]

  session = tf.Session()
  model_file = tf.train.latest_checkpoint(MODEL_SAVE_FOLD)
  saver.restore(session, model_file)
  print('Load model file from ' + model_file)

  logits_results = session.run(logits, feed_dict={x: X_test, y: y_test, prob: 1.0})
  softmax_probabilities = session.run(tf.nn.softmax(logits_results))
  predictions = session.run(tf.argmax(logits_results, 1))
  conv1_x_results, conv2_x_results = session.run(
    [conv1_x, conv2_x], feed_dict={x: X_test, y: y_test, prob: 1.0})

  os.system('rm -f ' + test_image_dir + 'conv*.png')
  outputFeatureMap(y_test, conv1_x_results, test_image_dir + 'conv1')
  outputFeatureMap(y_test, conv2_x_results, test_image_dir + 'conv2')
  for i in xrange(len(predictions)):
    s = softmax_probabilities[i]
    sorted_index = sorted(range(len(s)), key=lambda k: s[k], reverse=True)[:10]
    sorted_props = sorted(s, reverse=True)[:10]
    print('Prediction of label {}, correct label is {}'.format(predictions[i], y_test[i]))
    print(sorted_props)
    print(sorted_index)


def process(train_op, test_op, test_images):
  X_train, y_train = load_training_data(expand = train_op)
  X_valid, y_valid = load_valid_data()
  X_test, y_test = load_testing_data()

  image_depth = X_train[0].shape[2]

  print_out_training_data_summary(X_train, y_train, X_valid, y_valid, X_test, y_test)

  x = tf.placeholder(tf.float32, (None, 32, 32, image_depth))
  y = tf.placeholder(tf.int32, (None))
  one_hot_y = tf.one_hot(y, 43)

  prob = tf.placeholder_with_default(1.0, shape=())
  logits, regularizer, conv1_x, conv2_x = LeNet(x, dropout_prob = prob, image_depth = image_depth)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
  loss_operation = tf.reduce_mean(cross_entropy + BETA * regularizer)
  optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
  training_operation = optimizer.minimize(loss_operation)

  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
  accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  saver = tf.train.Saver(max_to_keep=EPOCHS)

  if train_op:
    train(X_train, y_train, X_valid, y_valid, X_test, y_test,
          x, y, prob, training_operation, loss_operation, accuracy_operation, saver)
  if test_op:
    check_test_accuracy(X_test, y_test, x, y, prob,
                        accuracy_operation, logits, saver)
  if test_images:
    test_real_images(x, y, prob, accuracy_operation, logits, conv1_x, conv2_x, saver)


def main():
  train_op, test_op, test_images = False, False, False
  if len(sys.argv) <= 1:
    train_op = True
    test_op = True
  elif sys.argv[1] == 'test':
    test_op = True
  elif sys.argv[1] == 'train':
    train_op = True
  elif sys.argv[1] == 'test_images':
    test_images = True
  else:
    print("Usage: python {} test|train|test_images".format(sys.argv[0]))
    exit()
  process(train_op, test_op, test_images)


if __name__ == "__main__":
  main()
