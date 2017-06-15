#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

learning_rate = 0.01
N = 100
N_steps = 300

# Training Data
train_X = np.linspace(-10, 10, N)
train_Y = 2*train_X + 3 + 5*np.random.random(N)

# Computational Graph
X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")
y = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_mean(tf.pow(y-Y, 2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(N_steps):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

        cost_val, W_val, b_val = sess.run([cost, W, b], feed_dict={X: train_X, Y:train_Y})
        print("step", step, "cost", cost_val, "w", W_val, "b", b_val)

plt.plot(train_X, train_Y, 'r*')
plt.plot(train_X, W_val*train_X+b_val, 'b-')
plt.show()