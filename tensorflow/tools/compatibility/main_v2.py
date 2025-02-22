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
tf.compat.v1.disable_v2_behavior()

print("\nNum GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Tensorflow version: ", tf.__version__)

# Formula
# y = Wx + b

# Training data, given x_train as inputs, we expect y_train as outputs
x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [-1.0, -2.0, -3.0, -4.0]

# Graph construction
# W and b are variables that our model will change
W = tf.Variable(initial_value=[1.0], dtype=tf.float32)
b = tf.Variable(initial_value=[1.0], dtype=tf.float32)

# x is an input placeholder and y is a placeholder used to tell model what correct answers are
x = tf.compat.v1.placeholder(dtype=tf.float32)
y_input = tf.compat.v1.placeholder(dtype=tf.float32)

# y_output is the formula we are trying to follow to produce an output given input from x
y_output = W * x + b

# Loss function and optimizer aim to minimize the difference between actual and expected outputs (total sums)
loss = tf.reduce_sum(input_tensor=tf.square(x=y_output - y_input))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = optimizer.minimize(loss=loss)

# Sessions are used to evaluate the tensor value of a node or nodes
session = tf.compat.v1.Session()
session.run(tf.compat.v1.global_variables_initializer())

# Total loss before training
print("\nTotal kerugian sebelum training:", session.run(fetches=loss, feed_dict={x: x_train, y_input: y_train}))

# Training phase, run the train step 1000 times
for _ in range(1000):
    session.run(fetches=train_step, feed_dict={x: x_train, y_input: y_train})

# Total loss and modified W and b values after training
print("Total kerugian setelah training:", session.run(fetches=[loss, W, b], feed_dict={x: x_train, y_input: y_train}))

# Test the model with some new values
print("Test kerugian dengan model baru:", session.run(fetches=y_output, feed_dict={x: [5.0, 10.0, 15.0]}))
