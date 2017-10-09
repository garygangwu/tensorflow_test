import sys
import time
from sklearn.utils import shuffle
import tensorflow as tf
from traffic_sign_data_loader import *
from traffic_sign_training_utils import *

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
  import matplotlib.image as mpimg
  session = tf.Session()
  model_file = tf.train.latest_checkpoint(MODEL_SAVE_FOLD)
  print('Load model file from ' + model_file)
  saver.restore(session, model_file)

  test_accuracy, logits_results = session.run(
    [accuracy_operation, logits],
    feed_dict={x: X_test, y: y_test, prob: 1.0})

  predictions = session.run(tf.argmax(logits_results, 1))
  for i in range(len(predictions)):
    if predictions[i] != y_test[i]:
      filename = 'images/bad_pred_' + str(y_test[i]) + '_' + str(predictions[i]) + '.png'
      mpimg.imsave(filename, X_test[i])

  print('')
  print("Test Accuracy = {:.3f}".format(test_accuracy))


def process(train_op, test_op):
  X_train, y_train = load_training_data(expand = train_op)
  X_valid, y_valid = load_valid_data()
  X_test, y_test = load_testing_data()

  image_depth = X_train[0].shape[2]

  print_out_training_data_summary(X_train, y_train, X_valid, y_valid, X_test, y_test)

  x = tf.placeholder(tf.float32, (None, 32, 32, image_depth))
  y = tf.placeholder(tf.int32, (None))
  one_hot_y = tf.one_hot(y, 43)

  prob = tf.placeholder_with_default(1.0, shape=())
  logits, regularizer = LeNet(x, dropout_prob = prob, image_depth = image_depth)
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


def main():
  train_op, test_op = False, False
  if len(sys.argv) <= 1:
    train_op = True
    test_op = True
  elif sys.argv[1] == 'test':
    test_op = True
  elif sys.argv[1] == 'train':
    train_op = True
  else:
    print("Usage: python {} test|train".format(sys.argv[0]))
    exit()
  process(train_op, test_op)


if __name__ == "__main__":
  main()
