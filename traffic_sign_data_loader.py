import numpy as np
import pickle

PREFIX = './data/'
TRAINING_FILE = PREFIX + 'train.p'
VALID_FILE = PREFIX + 'valid.p'
TESTING_FILE = PREFIX + 'test.p'

def load_training_data(expand = True):
  train = pickle.load(open(TRAINING_FILE, mode='rb'))
  X_train, y_train = train['features'], train['labels']
  image_depth = X_train[0].shape[2]

  if expand:
    X_train, y_train = expand_training_set(X_train, y_train, depth=image_depth)
  return normalize_images(X_train), y_train

def load_valid_data():
  valid = pickle.load(open(VALID_FILE, mode='rb'))
  X_valid, y_valid = valid['features'], valid['labels']
  return normalize_images(X_valid), y_valid

def load_testing_data():
  test = pickle.load(open(TESTING_FILE, mode='rb'))
  X_test, y_test = test['features'], test['labels']
  return normalize_images(X_test), y_test

def left_shift(a):
  x = np.roll(a,-1,axis=1)
  x[:, len(x[0])-1] = 0
  return x

def right_shift(a):
  x = np.roll(a,1,axis=1)
  x[:,0] = 0
  return x

def up_shift(a):
  x = np.roll(a,-1,axis=0)
  x[len(x)-1, :] = 0
  return x

def down_shift(a):
  x = np.roll(a,1,axis=0)
  x[0, :] =0
  return x

def expand_training_set(X_train, y_train, depth=3):
  X_train_left_shift = np.zeros((len(X_train), 32,32,depth))
  X_train_right_shift = np.zeros((len(X_train), 32,32,depth))
  X_train_up_shift = np.zeros((len(X_train), 32,32,depth))
  X_train_down_shift = np.zeros((len(X_train), 32,32,depth))
  X_train_left_down_shift = np.zeros((len(X_train), 32,32,depth))
  X_train_right_up_shift = np.zeros((len(X_train), 32,32,depth))
  X_train_left_up_shift = np.zeros((len(X_train), 32,32,depth))
  X_train_right_down_shift = np.zeros((len(X_train), 32,32,depth))

  for i in range(len(X_train)):
    image = X_train[i]
    X_train_left_shift[i] = left_shift(image)
    X_train_right_shift[i] = right_shift(image)
    X_train_up_shift[i] = up_shift(image)
    X_train_down_shift[i] = down_shift(image)
    X_train_left_down_shift[i] = left_shift(X_train_left_shift[i])
    X_train_right_up_shift[i] = right_shift(X_train_right_shift[i])
    X_train_left_up_shift[i] = up_shift(X_train_up_shift[i])
    X_train_right_down_shift[i] = down_shift(X_train_down_shift[i])

  return np.concatenate(
              (X_train,
               X_train_left_down_shift,
               X_train_right_up_shift,
               X_train_left_up_shift,
               X_train_right_down_shift,
               X_train_left_shift,
               X_train_right_shift,
               X_train_up_shift,
               X_train_down_shift),
              axis=0), \
         np.concatenate(
              (y_train,
               y_train,
               y_train,
               y_train,
               y_train,
               y_train,
               y_train,
               y_train,
               y_train),
              axis=0)

def normalize_images(images):
  return (images * 1.0 - 128) / 128

