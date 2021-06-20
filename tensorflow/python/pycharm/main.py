# This is a sample of Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os, sys, platform
print("\nPlatform:", platform.platform())
print("Python version: ", sys.version)
os.system('pip --version')
print("System path: ", os.environ.get('PATH'))

#Migrate your TensorFlow 1 code to TensorFlow 2
#https://www.tensorflow.org/guide/migrate

import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

print("\nNum GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Tensorflow version: ", tf.__version__, "\n")

# Formula
# y = Wx + b

# Training data, given x_train as inputs, we expect y_train as outputs
#x_train = [1.0, 2.0, 3.0, 4.0]
#y_train = [-1.0, -2.0, -3.0, -4.0]

x_train = tf.constant([1.0, 2.0, 3.0, 4.0])
y_train = tf.constant([-1.0, -2.0, -3.0, -4.0])

print("x_train = ", x_train)
print("y_train = ", y_train, "\n")

# Graph construction
# W and b are variables that our model will change
#W = tf.Variable(initial_value=[1.0], dtype=tf.float32)
#b = tf.Variable(initial_value=[1.0], dtype=tf.float32)
W = tf.Variable(tf.zeros(shape=(1,)), name="W")
b = tf.Variable(tf.ones(shape=(1,)), name="b")

print("W = ", W)
print("b = ", b, "\n")

#x is an input placeholder and y is a placeholder used to tell model what correct answers are
#x = tf.placeholder(dtype=tf.float32)
#print("x = ", x)
#y_input = tf.placeholder(dtype=tf.float32)
#print("y_input = ", y_input, "\n")

# y_output is the formula we are trying to follow to produce an output given input from x
#y_output = W * x + b

@tf.function
def forward(x):
  return W * x + b

y_output = forward(x_train)
print("y_output = ", y_output)

