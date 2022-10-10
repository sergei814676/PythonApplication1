from pyexpat import model
from telnetlib import SE
import metods
import PythonApplication1
import class_network_files
import keras
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM, Conv1D, Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras import layers
import h5py
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os.path
from tkinter import messagebox

class working_class(object):


    def __init__(self):


        self.Model_main = self.NewOneNetwork()
        print(self.Model_main)
        self.train = 0
        """Constructor"""
        #self.model = keras.models
        
        #ax = self.fig.add_subplot(111)
        #self.fig.clear()

          

          
       

        
    def Graph_clas (self):
       
          if (os.path.exists(PythonApplication1.message.get())):
           xx, yy = metods.file_acceptance()
          
           metods.network_plot(self.Model_main, xx, yy, self.Model_main)
          else:
              messagebox.showinfo('Информация', 'Выберите файл данных!')

       
       
    def NewOneNetwork (self): #


        #x,y=metods.file_acceptance()

        model = Sequential()
       # x_rows, x_cols = x.shape
        model.add(BatchNormalization())
        model.add(Dense(1280, activation='selu'))
        
        model.add(BatchNormalization())
        model.add(Dense(640, activation='selu'))
        model.add(BatchNormalization())
     #   model.add(Dense(1000, activation='relu'))
      #  model.add(Dense(640, activation='selu'))
      #  model.add(BatchNormalization())

        model.add(Dense(1,'elu'))

        model.compile(loss='mean_squared_error', optimizer='Ftrl')
        #history, model=fit_model(model,x, y)
        self.train = 0
        
        return  model

  
    def fit_models (self): 

        
        x, y = metods.file_acceptance()
        
       # self.history, self.Model_main=metods.fit_model(self.Model_main,self.x, self.y)
        self.train = 1
        #return history, model

        self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=500)
        self.mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)




        self.history = self.Model_main.fit(x, y, epochs=int(PythonApplication1.message_entry_number_of_epochs.get()), batch_size=40, shuffle=True, validation_split=(int(PythonApplication1.message_entry_test_percentage.get())//100),validation_freq=2, callbacks=[self.es, self.mc])
    





    pass




