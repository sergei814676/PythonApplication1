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
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, Flatten
import time, os, argparse, io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
dir = os.path.dirname(os.path.realpath(__file__))
tf.compat.v1.disable_eager_execution()

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
 
 def model(X, w):
    return tf.multiply(X, w) 
 
    
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

 #xxx=df['val'].to_numpy()
 #yyy=df['vale'].to_numpy()

 #trX = np.array([df['val'].to_numpy()], dtype=float)
 #trY = np.array([df['vale'].to_numpy()], dtype=float)
 
 """
 trX = np.linspace(-1, 1, 101)

 for i in range(num_coeffs):
    
    trY += trY_coeffs[i] * np.power(trX, i)
 trY += np.random.randn(*trX.shape) * 1.5
 """
 

 #trX=df['val'].to_numpy()
 #trY=df['vale'].to_numpy()

 trX = np.array(df['val'].to_numpy(), dtype=float)
 trY = np.array(df['vale'].to_numpy(), dtype=float)

 plt.scatter(trX, trY)


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
 url='E:/data2.csv'
 #url=PythonApplication1.message.get()
 dff = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)
 #x_data = np.linspace (-0.5, 0.5, 200) [:, np.newaxis] # ������������ 200 �����, ���������� �������������� �� -0,5 �� 0,5, �������� ������� �� 200 ����� � ������ �������
 #noise = np.random.normal (0, 0.02, x_data.shape) # ��������� ���������� ����
 #y_data = np.square(x_data) + noise  # y = x^2 + noise

 #x_data=df['val'].to_numpy()
 #y_data=df['vale'].to_numpy()

 #x_data = np.array([df['val'].to_numpy()], dtype=float)
 #y_data = np.array([df['vale'].to_numpy()], dtype=float)
    
    
 #x = tf.placeholder (tf.float32, [None, 1]) # ���������� �����������
 #y = tf.placeholder(tf.float32, [None, 1])


 #x = -50 + np.random.random((25000,1))*100
# y = x**2

 
 x=dff['val'].to_numpy()
 y=dff['vale'].to_numpy()
# Define model
 
 few_neurons=len(x)
 model = Sequential()
 model.add(Dense(few_neurons,input_shape=(1,), input_dim=1, activation='tanh'))
 model.add(Dense(few_neurons, activation='tanh'))
 model.add(Dense(few_neurons, activation='relu'))

 #model.add(Dense(10, activation='relu'))
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer='adam')
 model.fit(x, y, epochs=2700, batch_size=2000)

# predictions = model.predict([10, 5, 200, 13])
 #print(predictions) # Approximately 100, 25, 40000, 169
 
 
 """
 plt.subplot(2, 1, 1)
 plt.scatter(x, y, s = 1)
 plt.title('y = $x^2$')
 plt.ylabel('Real y')

 plt.subplot(2, 1, 2)
 plt.scatter(x, model.predict(x), s = 1)
 plt.xlabel('x')
 plt.ylabel('Approximated y')
 plt.show()
 """


 plt.scatter(x, y)
 #xx=[]


 xx=np.arange(min(x), max(x), 0.1)
 plt.scatter(xx, model.predict(xx))
 plt.show()


def NewFourNetwork(): #сеть LSTM
 url='E:/data2.csv'
 #url=PythonApplication1.message.get()
 dff = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)
 #x_data = np.linspace (-0.5, 0.5, 200) [:, np.newaxis] # ������������ 200 �����, ���������� �������������� �� -0,5 �� 0,5, �������� ������� �� 200 ����� � ������ �������
 #noise = np.random.normal (0, 0.02, x_data.shape) # ��������� ���������� ����
 #y_data = np.square(x_data) + noise  # y = x^2 + noise

 #x_data=df['val'].to_numpy()
 #y_data=df['vale'].to_numpy()

 #x_data = np.array([df['val'].to_numpy()], dtype=float)
 #y_data = np.array([df['vale'].to_numpy()], dtype=float)
    
    
 #x = tf.placeholder (tf.float32, [None, 1]) # ���������� �����������
 #y = tf.placeholder(tf.float32, [None, 1])


 #x = -50 + np.random.random((25000,1))*100
# y = x**2
 data_dim = 16

 timesteps = 8

 num_classes = 10

 batch_size = 32
 
 x=dff['val'].to_numpy()
 y=dff['vale'].to_numpy()
# Define model
 
 few_neurons=len(x)
 model = Sequential()

 model.add(Dense(few_neurons, input_shape=(1,), input_dim=1, activation='tanh'))
 model.add(LeakyReLU(alpha=0.2))
 model.add(BatchNormalization(momentum=0.8))
 model.add(Dense(few_neurons*2))
 model.add(LeakyReLU(alpha=0.2))
 model.add(BatchNormalization(momentum=0.8))
 model.add(Dense(few_neurons*4))
 model.add(LeakyReLU(alpha=0.2))
 model.add(BatchNormalization(momentum=0.8))
 model.add(Dense(few_neurons, activation='relu'))

 model.add(Dense(1))
 model.summary()
 model.compile(loss='mean_squared_error', optimizer='adam')
 model.fit(x, y, epochs=20, batch_size=200)
 plt.scatter(x, y)
 #xx=[]


 xx=np.arange(min(x), (max(x)+max(x)//2), 0.1)
 plt.scatter(xx, model.predict(xx), s=1)
 plt.show() 



def NewTwoNetwork (): #ФУНКЦИЯ,из за которой возможно  ошибка, с выкидыванием
   #  x = np.arange(0, 2*np.pi, 2*np.pi/1000).reshape((1000,1))
    # y = np.sin(x)
    # plt.plot(x,y)
    # plt.show()
 url=PythonApplication1.message.get()
 #df = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)
 #x = np.array([df['val'].to_numpy()], dtype=float)
 #y = np.array([df['vale'].to_numpy()], dtype=float)
 data = pd.read_csv(url)

# Drop date variable
 #data = data.drop(['DATE'], 1)

# Dimensions of dataset
 n = data.shape[0]
 p = data.shape[1]

# Make data a np.array
 data = data.values

# Training and test data
 train_start = 0
 train_end = int(np.floor(0.8*n))
 test_start = train_end + 1
 test_end = n
 data_train = data[np.arange(train_start, train_end), :]
 data_test = data[np.arange(test_start, test_end), :]

# Scale data
 scaler = MinMaxScaler(feature_range=(-1, 1))
 scaler.fit(data_train)
 data_train = scaler.transform(data_train)
 data_test = scaler.transform(data_test)

# Build X and y
 X_train = data_train[:, 1:]
 y_train = data_train[:, 0]
 X_test = data_test[:, 1:]
 y_test = data_test[:, 0]

# Number of stocks in training data
 n_stocks = X_train.shape[1]

# Neurons
 n_neurons_1 = 1024
 n_neurons_2 = 512
 n_neurons_3 = 256
 n_neurons_4 = 128

# Session
 net = tf.InteractiveSession()

# Placeholder
 X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
 Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
 sigma = 1
 weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
 bias_initializer = tf.zeros_initializer()

# Hidden weights
 W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
 bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
 W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
 bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
 W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
 bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
 W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
 bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output weights
 W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
 bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
 hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
 hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
 hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
 hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (transpose!)
 out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
 mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
 opt = tf.train.AdamOptimizer().minimize(mse)

# Init
 net.run(tf.global_variables_initializer())

# Setup plot
 plt.ion()
 fig = plt.figure()
 ax1 = fig.add_subplot(111)
 line1, = ax1.plot(y_test)
 line2, = ax1.plot(y_test * 0.5)
 plt.show()

# Fit neural net
 batch_size = 256
 mse_train = []
 mse_test = []

# Run
 epochs = 10
 for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.pause(0.01)

def NewThreeNetwork ():
 url=PythonApplication1.message.get()
 dff = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)


 x=dff['val'].to_numpy()
 t=dff['vale'].to_numpy()
 #learning_rate = 0.01
 #training_epochs = 40

 tf.disable_v2_behavior()
 def convertToMatrix(data, step):
  X, Y =[], []
  for i in range(len(data)-step):
   d=i+step
   X.append(data[i:d,])
   Y.append(data[d,])
  return np.array(X), np.array(Y)

 step = 2
 N = len(x)
 Tp =(len(x)//4)*3
 #step = 4
 #N = 1000    
 #Tp = 800
 #t=np.arange(0,N)
 #x=np.sin(0.02*t)+2*np.random.rand(N)

 df = pd.DataFrame(x)
 df.head()

 plt.plot(t, x,'*')
#plt.show()

 values=df.values
 train,test = values[0:Tp,:], values[Tp:N,:]

# add step elements into train and test
 test = np.append(test,np.repeat(test[-1,],step))
 train = np.append(train,np.repeat(train[-1,],step))
 
 trainX,trainY =convertToMatrix(train,step)
 testX,testY =convertToMatrix(test,step)
 trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
 testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

 model = Sequential()
 model.add(SimpleRNN(units=32, input_shape=(1,step), activation="relu"))
 model.add(Dense(8, activation="relu")) 
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer='rmsprop')
 model.summary()

 model.fit(trainX,trainY, epochs=100, batch_size=16, verbose=2)
 trainPredict = model.predict(trainX)
 testPredict= model.predict(testX)
 predicted=np.concatenate((trainPredict,testPredict),axis=0)

 trainScore = model.evaluate(trainX, trainY, verbose=0)
 print(trainScore)

 index = df.index.values
# plt.plot(df)
 plt.plot(index,predicted)
 plt.axvline(df.index[Tp], c="r")
 plt.show()






def evaluation_training():


















def choiceCombobox():#выбор функции
     if PythonApplication1.comboExample.get() == "probanerset":
             probanerset()
     if PythonApplication1.comboExample.get() == "Lineq":
             postroenie()
     if PythonApplication1.comboExample.get() == "Polinel":
             postroeniepolin()
     if PythonApplication1.comboExample.get() == "NewOneNetwork":
             NewOneNetwork()
     if PythonApplication1.comboExample.get() == "NewTwoNetwork":
             NewTwoNetwork()
     if PythonApplication1.comboExample.get() == "NewThreeNetwork":
             NewThreeNetwork()
     if PythonApplication1.comboExample.get() == "NewFourNetwork":
             NewFourNetwork()