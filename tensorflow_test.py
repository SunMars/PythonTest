
import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
import cv2
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cap = cv2.VideoCapture(0)

# 准备训练数据，假设其分布大致符合 y = 1.2x + 0.0

n_train_samples = 200

X_train = np.linspace(-5, 5, n_train_samples)
Y_train = 1.2*X_train + np.random.uniform(-1.0, 1.0, n_train_samples)  # 加一点随机扰动


# 准备验证数据，用于验证模型的好坏
n_test_samples = 50
X_test = np.linspace(-5, 5, n_test_samples)
Y_test = 1.2*X_test


# 参数学习算法相关变量设置
learning_rate = 0.01
batch_size = 20
summary_dir = 'logs'

print('~~~~~~~~~~开始设计计算图~~~~~~~~')

# 使用 placeholder 将训练数据/验证数据送入网络进行训练/验证
# shape=None 表示形状由送入的张量的形状来确定
with tf.name_scope('Input'):
    X = tf.placeholder(dtype=tf.float32, shape=None, name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=None, name='Y')


# 决策函数(参数初始化)
with tf.name_scope('Inference'):
    W = tf.Variable(initial_value=tf.truncated_normal(shape=[1]), name='weight')
    b = tf.Variable(initial_value=tf.truncated_normal(shape=[1]), name='bias')
    Y_pred = tf.multiply(X, W) + b


# 损失函数(MSE)
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.square(Y_pred - Y), name='loss')
    tf.summary.scalar('loss', loss)


# 参数学习算法(Mini-batch SGD)
with tf.name_scope('Optimization'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# 初始化所有变量
init = tf.global_variables_initializer()


# 汇总记录节点
merge = tf.summary.merge_all()


# 开启会话，进行训练
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logdir=summary_dir, graph=sess.graph)

    while True:
        ret, image_np = cap.read()
        print("camera capture!")
        cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    for i in range(201):
        j = np.random.randint(0, 10)  # 总共200训练数据，分十份[0, 9]
        X_batch = X_train[batch_size*j: batch_size*(j+1)]
        Y_batch = Y_train[batch_size*j: batch_size*(j+1)]

        _, summary, train_loss, W_pred, b_pred = sess.run([optimizer, merge, loss, W, b], feed_dict={X: X_batch, Y: Y_batch})
        test_loss = sess.run(loss, feed_dict={X: X_test, Y: Y_test})

        # 将所有日志写入文件
        summary_writer.add_summary(summary, global_step=i)
        print('step:{}, losses:{}, test_loss:{}, w_pred:{}, b_pred:{}'.format(i, train_loss, test_loss, W_pred[0], b_pred[0]))

        if i == 200:
            # plot the results
            plt.plot(X_train, Y_train, 'bo', label='Train data')
            plt.plot(X_test, Y_test, 'gx', label='Test data')
            plt.plot(X_train, X_train * W_pred + b_pred, 'r', label='Predicted data')
            plt.legend()
            plt.show()

    summary_writer.close()
