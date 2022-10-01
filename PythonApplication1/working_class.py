from pyexpat import model
from telnetlib import SE
import metods
import PythonApplication1
import class_network_files
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM, Conv1D, Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras import layers
import h5py



class working_class(object):


    def __init__(self):

        self.Model_main = self.NewOneNetwork()
        self.train = 0
        """Constructor"""
        #self.model = keras.models
        
        #ax = self.fig.add_subplot(111)
        #self.fig.clear()
        
    def training_clas (self, model, x, y):
       
           self.history, self.model=self.fit_model(model,x, y)
          

          
       

        
    def Graph_clas (self, model, x, y,history):
       

           metods.network_plot(model, x, y,history)
       
       
    def NewOneNetwork (self): #создание скелета модели


        #x,y=metods.file_acceptance()

        model = Sequential()
       # x_rows, x_cols = x.shape
        model.add(Dense(300, activation='elu')) 
        #model.add(Dense(600, activation='linear'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(400, activation='tanh'))

        model.add(Dense(1,'elu'))

        model.compile(loss='mean_squared_error', optimizer='RMSprop')
        #history, model=fit_model(model,x, y)
        self.train = 0
        return  model

  
    def fit_model (self, model,x, y): 


        
        history, model=metods.fit_model(model,x, y)
        self.train = 1
        return history, model

    pass




