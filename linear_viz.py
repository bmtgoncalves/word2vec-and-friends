#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensorflow as tf

def plot_frame(train_X, train_Y, W_val, b_val, step, cost, Ws, bs, costs):
    fig = plt.figure(figsize=(11, 8.5)) 
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1]) 

    plt.subplot(gs[:, 0])
    plt.plot(train_X, train_Y, color="dodgerblue", marker="*", linestyle="none")
    plt.plot(train_X, W_val*train_X+b_val, '-', color='blueviolet', linestyle='solid')
    plt.legend(["data","y=%g x+%g" % (W_val, b_val)])
    plt.title("step: %u cost: %g" % (step, cost_val))
    plt.xlim(-10, 10)
    plt.ylim(-20, 30)

    plt.subplot(gs[0, 1])
    plt.plot(range(len(Ws)), Ws, color="green", linestyle="solid")
    plt.xlim(0, 120)
    plt.ylim(-10, 10)
    plt.xlabel("step")
    plt.ylabel("W")

    plt.subplot(gs[1, 1])
    plt.plot(range(len(bs)), bs, color="green", linestyle="solid")
    plt.xlim(0, 120)
    plt.ylim(-10, 30)
    plt.xlabel("step")
    plt.ylabel("b")

    plt.subplot(gs[2, 1])
    plt.semilogy(range(len(costs)), costs, color="green", linestyle="solid")
    plt.xlim(0, 120)
    plt.ylim(1, 10000)
    plt.xlabel("step")
    plt.ylabel("cost")

    plt.tight_layout()
    plt.savefig("frame_%03u.png" % step, dpi=300)
    plt.close()

learning_rate = 0.022
N = 100
N_steps = 120

# Training Data
train_X = np.linspace(-10, 10, N)
train_Y = 2*train_X + 3 + (2*np.random.random(N)-1)

# Computational Graph
X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(-10., name="weight", dtype="float")
b = tf.Variable(30., name="bias", dtype="float")
y = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_mean(tf.pow(y-Y, 2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

Ws = [-10]
bs = [30]
costs = []

with tf.Session() as sess:
    sess.run(init)

    cost_val, W_val, b_val = sess.run([cost, W, b], feed_dict={X: train_X, Y:train_Y})
    costs.append(cost_val)
    plot_frame(train_X, train_Y, W_val, b_val, 0, cost, Ws, bs, costs)

    for step in range(N_steps):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

        cost_val, W_val, b_val = sess.run([cost, W, b], feed_dict={X: train_X, Y:train_Y})

        Ws.append(W_val)
        bs.append(b_val)
        costs.append(cost_val)

        print("step", step+1, "cost", cost_val, "w", W_val, "b", b_val)
        plot_frame(train_X, train_Y, W_val, b_val, step, cost, Ws, bs, costs)

