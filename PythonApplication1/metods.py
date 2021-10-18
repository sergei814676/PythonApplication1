import matplotlib
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import math, random
import pandas as pd
import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tkinter import *
import tkinter as tk
import matplotlib
from PythonApplication1 import matplotlib, plt, plt
import PythonApplication1
from tkinter import *
import tkinter as tk 
from tkinter import messagebox
import tkinter.filedialog as fd
from tkinter import ttk
import metods

n_nodes_hl1 = 100
n_nodes_hl2 = 100

def postroenie ():#��������
 #url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQ_Mv5MB1Ty1ixEnsXeDJwz0_31rQ1woLrfvmT-SSRuqcUybp6cTy8CgqncjtYxC41ZG5HlzPqUN0au/pub?gid=0&single=true&output=csv'
 
 url=PythonApplication1.message.get()
 df = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)


 xxx=df['val'].to_numpy()
 yyy=df['vale'].to_numpy()

 xx = np.array([df['val'].to_numpy()], dtype=float)
 yy = np.array([df['vale'].to_numpy()], dtype=float)
 # Define the number of nodes
 n_nodes_hl1 = 100
 n_nodes_hl2 = 100


 # Define the number of outputs and the learn rate
 n_classes = 1
 learn_rate = 0.1

 xx.dtype
 learning_rate = 0.01
 training_epochs = 100

 x_train=df['val'].to_numpy()
 y_train=df['vale'].to_numpy()


 X = tf.placeholder(tf.float32)
 Y = tf.placeholder(tf.float32)
 
 
    
 w = tf.Variable(0.0, name="weights")
 
 y_model = model(X, w)
 cost = tf.square(Y-y_model)
 
 train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
 
 sess = tf.Session()
 init = tf.global_variables_initializer()
 sess.run(init)
    
 for epoch in range(training_epochs):
   for (x, y) in zip(x_train, y_train):
     sess.run(train_op, feed_dict={X: x, Y: y})
 
 w_val = sess.run(w)
 
 sess.close()
 plt.scatter(x_train, y_train)
 y_learned = x_train*w_val
 plt.plot(x_train, y_learned, 'r')
 plt.show() 




def postroeniepolin (): #������� ������ �� ��������
 #url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQ_Mv5MB1Ty1ixEnsXeDJwz0_31rQ1woLrfvmT-SSRuqcUybp6cTy8CgqncjtYxC41ZG5HlzPqUN0au/pub?gid=0&single=true&output=csv'
 tf.disable_v2_behavior()  
 url=PythonApplication1.message.get()
 df = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)
 num_coeffs=6
 trY_coeffs= [1, 2, 3, 4, 5, 6]
 learning_rate = 0.01
 training_epochs = 40
 trY=0
 xxx=df['val'].to_numpy()
 yyy=df['vale'].to_numpy()

 #trX = np.array([df['val'].to_numpy()], dtype=float)
 #trY = np.array([df['vale'].to_numpy()], dtype=float)
 
 
 trX = np.linspace(-1, 1, 101)

 for i in range(num_coeffs):
    
    trY += trY_coeffs[i] * np.power(trX, i)
 trY += np.random.randn(*trX.shape) * 1.5
 plt.scatter(trX, trY)

 """
 trX=df['val'].to_numpy()
 trY=df['vale'].to_numpy()
 """


 X = tf.placeholder(tf.float32)
 Y = tf.placeholder(tf.float32)





 def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)
 
 w = tf.Variable([0.] * num_coeffs, name="parameters")
 y_model = model(X, w)
  
 cost = (tf.pow(Y-y_model, 2))
 train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
 
 sess = tf.Session()
 init = tf.global_variables_initializer()
 sess.run(init)
 
 for epoch in range(training_epochs):
    for (x, y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})
 
 w_val = sess.run(w)
 print(w_val)
 sess.close()
 plt.scatter(trX, trY)
 trY2 = 0
 for i in range(num_coeffs):
    trY2 += w_val[i] * np.power(trX, i)


 plt.plot(trX, trY2, 'r')
 plt.show()


class SeriesPredictor:
    def __init__(self, input_dim, seq_size, hidden_dim=10):
 
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim
 
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])
 
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
 
        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out


    def test(self, test_x):
       with tf.Session() as sess:
           tf.get_variable_scope().reuse_variables()
           self.saver.restore(sess, './model.ckpt')
           output = sess.run(self.model(), feed_dict={self.x: test_x})
           print(output)

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(1000):                           
                _, mse = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if i % 100 == 0:
                    print(i, mse)
            save_path = self.saver.save(sess, 'model.ckpt')
            print('Model saved to {}'.format(save_path))



def probanerset ():
 #if __name__ == '__main__':
    predictor = SeriesPredictor(input_dim=1, seq_size=4, hidden_dim=10)
    train_x = [[[1], [2], [5], [6]],
               [[5], [7], [7], [8]],
               [[3], [4], [5], [7]]]
    train_y = [[1, 3, 7, 11],
               [5, 12, 14, 15],
               [3, 7, 9, 12]]
    predictor.train(train_x, train_y)
    test_x = [[[1], [2], [3], [4]],
              [[4], [5], [6], [7]]]
    predictor.test(test_x)


def NewOneNetwork ():  #ФУНКЦИЯ,из за которой ошибка, но с 4 перезапуска вроде не вылетает
 url=PythonApplication1.message.get()
 df = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)
 x_data = np.linspace (-0.5, 0.5, 200) [:, np.newaxis] # ������������ 200 �����, ���������� �������������� �� -0,5 �� 0,5, �������� ������� �� 200 ����� � ������ �������
 noise = np.random.normal (0, 0.02, x_data.shape) # ��������� ���������� ����
 y_data = np.square(x_data) + noise  # y = x^2 + noise

 #x_data=df['val'].to_numpy()
 #y_data=df['vale'].to_numpy()

 #x_data = np.array([df['val'].to_numpy()], dtype=float)
 #y_data = np.array([df['vale'].to_numpy()], dtype=float)
    
    
 x = tf.placeholder (tf.float32, [None, 1]) # ���������� �����������
 y = tf.placeholder(tf.float32, [None, 1])

# ���������� ������� ����
 weight_1 = tf.Variable (tf.random_normal ([1, 10])) # ������� ����� 1 * 10, �� ���� 1 ����, 10 ������������� �����
 biase_1 = tf.Variable (tf.zeros ([1, 10])) # �������� ��������
 wx_plus_1 = tf.matmul (x, weight_1) + biase_1 # ������� ������ � ��� ����������
 L1 = tf.nn.tanh (wx_plus_1) # ������� ���������
 
 # ���������� �������� ����
 weight_2 = tf.Variable (tf.random_normal ([10, 1])) # 10 ������������� �����, 1 �������� ����
 biase_2 = tf.Variable(tf.zeros([1, 1]))
 wx_plus_2 = tf.matmul(L1, weight_2) + biase_2
 prediction = tf.nn.tanh(wx_plus_2)
 
 loss = tf.reduce_mean(tf.square (y-prediction)) # ������� ������� ������ ��� ������ ������, � ����� ������� �������
 train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # ����������� ������ ������������ ������, �������� �������� 0.1, ��� ������������ ������
 with tf.Session() as sees:
  sees.run (tf.global_variables_initializer ()) # ������������� ����������
 for i in range(2000):
  sees.run(train_step, feed_dict={x:x_data,y:y_data}) # ��������� 2000 ����������
 
 prediction_value = see.run (prediction, feed_dict = {x: x_data}) # �������
 plt.figure () # �������
 plt.scatter (x_data, y_data) # �������� ����� �����
 plt.plot (x_data, prediction_value, 'r-', lw = 5) # �������� ������ ��������
 plt.show()



n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_classes = 1
learn_rate = 1.0e-9
y_ph = tf.placeholder('float')
def NewTwoNetwork (): #ФУНКЦИЯ,из за которой возможно  ошибка, с выкидыванием
     x = np.arange(0, 2*np.pi, 2*np.pi/1000).reshape((1000,1))
     y = np.sin(x)
    # plt.plot(x,y)
    # plt.show()

# Define the number of nodes
     n_nodes_hl1 = 100
     n_nodes_hl2 = 100

    # Define the number of outputs and the learn rate
     n_classes = 1
     learn_rate = 0.1

    # Define input / output placeholders
     x_ph = tf.placeholder('float', [None, 1])
     y_ph = tf.placeholder('float')
    # Routine to compute the neural network (2 hidden layers)
     train_neural_network(x_ph)

def neural_network_model(data):
     hidden_1_layer = {'weights': tf.Variable(tf.random_normal([1, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

     hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

     output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}


    # (input_data * weights) + biases
     l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
     l1 = tf.nn.relu(l1)

     l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
     l2 = tf.nn.relu(l2)

     output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])

     return output

def train_neural_network(x_ph):
     prediction = neural_network_model(x_ph)
     cost = tf.reduce_mean(tf.square(prediction - y_ph))
     optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

    # cycles feed forward + backprop
     hm_epochs = 10

     with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Train in each epoch with the whole data
        for epoch in range(hm_epochs):
            epoch_loss = 0
            _, c = sess.run([optimizer, cost], feed_dict = {x_ph: x, y_ph: y})
            epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy;', accuracy.eval({x_ph: x, y_ph: x}))




def choiceCombobox():#выбор функции
     if PythonApplication1.comboExample.get() == "probanerset":
             probanerset()
     if PythonApplication1.comboExample.get() == "Lineq":
             postroenie()
     if PythonApplication1.comboExample.get() == "Polinel":
             postroeniepolin()
     if PythonApplication1.comboExample.get() == "NewOneNetwork":
             NewOnewNetwork()
     if PythonApplication1.comboExample.get() == "NewTwoNetwork":
             NewTwoNetwork()