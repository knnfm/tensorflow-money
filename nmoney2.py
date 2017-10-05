# coding: utf-8

# 必要なモジュールを読み込む
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

# 1.予測式(モデル)を記述する
# 入力変数と出力変数のプレースホルダを生成
x = tf.placeholder(tf.float32, shape=(None, 3), name="x")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# モデルパラメータ
a = tf.Variable(tf.zeros((3, 1)), name="a")
# モデル式
y = tf.matmul(x, a)

# 2.学習に必要な関数を定義する
# 誤差関数(loss)
loss = tf.reduce_mean(tf.square(y_ - y))
# 最適化手段を選ぶ(最急降下法)
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

# 3.実際に学習処理を実行する
# (1)訓練データを生成する
train_x = np.array([[1., 2., 3.], [3., 2., 1.], [5., 6., 7.]])
train_y = np.array([190., 330., 660.]).reshape(3, 1)
print "x=", train_x
print "y=", train_y

# (2) セッションを準備し、変数を初期化
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# (3) 最急勾配法でパラメータ更新(100回更新する)
for i in range(100):
    _, l, a_ = sess.run([train_step, loss, a], feed_dict={x: train_x, y_: train_y})
    if (i + 1) % 10 == 0:
        print "step=%3d, a1=%6.2f, a2= %6.2f, loss = %.2f" % (i + 1, a_[0], a_[1], l)

# (4) 学習結果を出力
est_a = sess.run(a, feed_dict={x: train_x, y_:train_y})
print "Estimated: a1=%6.2f, a2=%6.2f" % (est_a[0], est_a[1])

# 4.新しいデータに対して予測する
# (1) 新しいデータを用意
new_x = np.array([2., 3., 4.]).reshape(1, 3)

# (2) 学習結果を使って、予測実施
new_y = sess.run(y, feed_dict={x: new_x})
print new_y

# 5.後片付け
# セッションを閉じる
sess.close()
