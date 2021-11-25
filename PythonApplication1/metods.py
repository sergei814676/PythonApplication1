import matplotlib
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import math, random
import pandas as pd
import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow.compat as tf
from tkinter import *
from tkinter import messagebox
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
from tkinter import messagebox as mb
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, Flatten
import time, os, argparse, io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras import layers
import h5py
from keras.layers import TimeDistributed
dir = os.path.dirname(os.path.realpath(__file__))
#tf.compat.v1.disable_eager_execution()

def postroenie ():#��������
 #url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQ_Mv5MB1Ty1ixEnsXeDJwz0_31rQ1woLrfvmT-SSRuqcUybp6cTy8CgqncjtYxC41ZG5HlzPqUN0au/pub?gid=0&single=true&output=csv'
 
 url=PythonApplication1.message.get()
 df = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)



 #xxx=df['val'].to_numpy()
 #yyy=df['vale'].to_numpy()

 #xx = np.array([df['val'].to_numpy()], dtype=float)
 #yy = np.array([df['vale'].to_numpy()], dtype=float)
 # Define the number of nodes
 n_nodes_hl1 = 100
 n_nodes_hl2 = 100


 # Define the number of outputs and the learn rate
 n_classes = 1
 

 #xx.dtype
 learning_rate = 0.01
 training_epochs = 100

 #x_train=np.array(df['val'].to_numpy(), dtype=float)
 #y_train=np.array(df['vale'].to_numpy(), dtype=float)

 x_train = np.linspace(-1, 1, 101)
 y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

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


def NewOneNetwork (): 
 #url='E:/data2.csv'
 url=PythonApplication1.message.get()
 dff = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)

 
 x=dff['val'].to_numpy()
 y=dff['vale'].to_numpy()
# Define model
 
 few_neurons=len(x)
 model = Sequential()
 initializer = keras.initializers.HeNormal(seed=None)
 model.add(Dense(512, input_dim=1, activation='tanh'))
 model.add(Dense(1024, activation='tanh'))
 model.add(Dense(512, activation='tanh')) 
 model.add(Dense(1, activation='elu'))

 sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
 model.compile(loss='mean_squared_error', optimizer='adam')
 model.fit(x, y, epochs=3000, batch_size=2000)

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
 plt.scatter(xx, model.predict(xx), s=1)
 plt.show()


def NewFourNetwork(): #сеть LSTM
 url='E:/data3.csv'
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
 model.add(Dense(few_neurons,input_shape=(1,), input_dim=1, activation='relu'))
 #model.add(Dense(few_neurons, activation='relu'))
 model.add(Dense(few_neurons*2))

 #model.add(Dense(10, activation='relu'))
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer='adam')
 model.fit(x, y, epochs=2700, batch_size=2000)





 plt.scatter(x, y)
 #xx=[]


 xx=np.arange(min(x), (max(y)+max(x)//2), 0.1)
 plt.scatter(xx, model.predict(xx), s=1)
 plt.show() 



def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)




def NewTwoNetwork (): #ФУНКЦИЯ,из за которой возможно  ошибка, с выкидыванием

 #url='E:/data3.csv'
 url=PythonApplication1.message.get()
 dff = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)

 x=dff['val'].to_numpy()
 y=dff['vale'].to_numpy()

# Dimensions of dataset
 model = tf.keras.Sequential(
  [
  tf.keras.layers.Dense(100, activation="relu"),
  tf.keras.layers.Dense(100, activation="relu"),
  tf.keras.layers.Dense(1, activation="linear"),
  gpflux.layers.GPLayer(
  kernel, inducing_variable, num_data=num_data,
  num_latent_gps=output_dim
  ),
  gpflux.layers.GPLayer(
  kernel, inducing_variable, num_data=num_data,
  num_latent_gps=output_dim
  ),
  likelihood_container,
  ]
 )
 model.fit(x_training, y_training, epochs=200, batch_size=512)
 plt.scatter(x, y)
 
 xx=np.arange(min(x), (max(x)+max(x)//2), 0.1)

 plt.scatter(xx, model.predict(xx), s=1)
 plt.show() 


def stable_network():
 model = Sequential()
 initializer = keras.initializers.HeNormal(seed=None)
 model.add(Dense(1, input_dim=1, activation='tanh'))
 model.add(Dense(1024, activation='tanh'))
 model.add(Dense(512, activation='tanh')) 
 model.add(Dense(1, activation='elu'))

 sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
 model.compile(loss='mean_squared_error', optimizer='adam')
 return model





def NewThreeNetwork ():
 url=PythonApplication1.message.get()
 dff = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)


 x=dff['val'].to_numpy()
 y=dff['vale'].to_numpy()
 


 model_one=stable_network()
 model_one.fit(x, y, epochs=100, batch_size=2000)
 model_one.save_weights('E:\my_model.h5')
 print(model_one.get_weights()[3])
 model_two = Sequential()
 #model_two.load_weights('E:\my_model.h5')
 initializer = keras.initializers.HeNormal(seed=None)
 model_two.add(Dense(1,activation='elu',kernel_initializer=set_weights(model_one.get_weights()[3])))

 model_two.add(Dense(1024, activation='tanh'))
 model_two.add(Dense(512, activation='tanh')) 
 model_two.add(Dense(1, activation='elu'))

 sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

 model_two.compile(loss='mean_squared_error', optimizer='adam')
 #model_two.load_weights('E:\my_model.h5', by_name=True)
 model_two.fit(x, y, epochs=500, batch_size=2000)
 


 plt.scatter(x, y)
 #xx=[]
 
 xx=np.arange(min(x)-10, (max(x)+max(x)//2), 0.1)
 plt.scatter(xx, model.predict(xx), s=1)
 plt.show() 








def conclusion_MSE(a): #выводит сообщение
        msg = a
        mb.showinfo("MSE", msg)

def PodborZnacheny():# тандем НС

 #url='E:/data3.csv'
 url=PythonApplication1.message.get()
 dff = pd.read_csv(url, names=['val','vale'],decimal='.', delimiter=',', dayfirst=True)

 fun_active_all=['relu','tanh','elu','softmax','selu','softplus','softsign','sigmoid','hard_sigmoid','linear'] 
 fun_optimizers_all=['RMSprop', 'sgd', 'adam', 'Nadam','Adamax','Adadelta', 'Adagrad']
 x_files=dff['val'].to_numpy()
 y_files=dff['vale'].to_numpy()
# Define model
 
 x=np.ones(len(x_files))
 y=np.ones(len(y_files))#Создание массивов

 x_y_shuffle=np.ones((2,len(x_files))) #Перетосовка точек 
 x_y_shuffle[0]=x_files.copy()
 x_y_shuffle[1]=y_files.copy()
 x_y_shuffle=x_y_shuffle.T
 np.random.shuffle(x_y_shuffle)
 x_y_shuffle=x_y_shuffle.T
 x=x_y_shuffle[0]
 y=x_y_shuffle[1]

 temp = list(zip(x_y_shuffle[0], x_y_shuffle[1]))


 separation_array=(len(x)//4)*3


 x_training1=np.split(x, [0, separation_array])
 y_training1=np.split(y, [0, separation_array])

 x_training=x_training1[1]
 y_training=y_training1[1]

 x_test=x_training1[2]
 y_test=y_training1[2]
 i=0
 min_index=1000
 min_element=1000
 search_min_mse=np.ones([101])
 while i<=100:
  search_min_mse[i]=1000
  i+=1

  
     
     
 i=0
 while i<=len(fun_active_all)-1:




  few_neurons=len(x)
  model = Sequential()
 # tens_i=i

  model.add(Dense(512,input_shape=(1,), input_dim=1, activation=fun_active_all[i]))
  model.add(Dense(1024, activation=fun_active_all[i]))
 # model.add(Dense(1024, activation=fun_active_all[i]))
  model.add(Dense(512, activation=fun_active_all[i]))
  model.add(Dense(1))

 #model.add(Dense(1))
 
 
 #model.add(Dense(200, activation='sigmoid'))
 #model.add(Dense(200, activation='sigmoid'))
 #model.add(Dense(100, activation='sigmoid'))
 #model.add(Dense(5, activation='sigmoid'))
 #model.add(Dense(10, activation='relu'))

  
  model.compile(loss='mean_squared_error', optimizer= 'RMSprop')

  model.fit(x_training, y_training, epochs=200+i, batch_size=512)

   #search_min_mse[i]=mean_squared_error(y_test, model.predict(x_test))
  if (mean_squared_error(y_test, model.predict(x_test))<min_element):
     min_element=mean_squared_error(y_test, model.predict(x_test))
     min_index=i
    
  i +=1
  

 mse_a=min_index

 model = Sequential()

 model.add(Dense(512,input_shape=(1,), input_dim=1, activation=fun_active_all[mse_a]))
 model.add(Dense(1024, activation=fun_active_all[mse_a]))
# model.add(Dense(1024, activation=fun_active_all[mse_a]))
 model.add(Dense(512, activation=fun_active_all[mse_a]))


 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer= 'RMSprop')

 model.fit(x_training, y_training, epochs=200, batch_size=200)

 plt.scatter(x_training, y_training)
 #xx=[]
 plt.scatter(x_test, y_test)
 
 conclusion_MSE(fun_active_all[min_index])

 xx=np.arange(min(x), (max(x)+max(x)//2), 0.1)

 plt.scatter(xx, model.predict(xx), s=1)
 plt.show() 














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
     if PythonApplication1.comboExample.get() == "NewFiveNetwork":
             NewFiveNetwork()
     if PythonApplication1.comboExample.get() == "PodborZnacheny":
             PodborZnacheny()