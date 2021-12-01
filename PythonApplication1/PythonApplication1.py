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


# This is the button callback function
# This must be visible to the button, so we must define it before the button widget!



def choose_file():
        filetypes =[('Файл необходимый', '*.xlsx *.csv'), ('Все файлы', '*')]
        dlg = fd.Open(filetypes = filetypes)
        fl = dlg.show()
        if fl != '':
            message_entry.delete(0, END)
            message_entry.insert(0, fl)


        #if filename:
          #  print(filename)
       # message_entry.insert(0, fl)


root = Tk()
root.title("Аппроксимация машинным обучением")
root.geometry("300x250")
 
message = StringVar()
 
message_entry = Entry(textvariable=message)
message_entry.place(relx=.5, rely=.1, anchor="c")

btn_file = Button(text="Выбрать файл", command=choose_file)
btn_file.place(relx=.8, rely=.1, anchor="c") 

message_button = Button(text="Запуск", command=metods.choiceCombobox)
message_button.place(relx=.5, rely=.5, anchor="c")

comboExample = ttk.Combobox(root, 
                            values=["PodborZnacheny",
                                    "NewOneNetwork", 
                                    "Polinel",
                                    "probanerset",
                                    "Lineq",
                                    "NewTwoNetwork",
                                    "NewThreeNetwork",
                                    "NewFourNetwork",
                                    "NewFiveNetwork"],
                            state="readonly")
comboExample.place(relx=.5, rely=.8, anchor="c")

comboExample.current(0)



tf.disable_v2_behavior()


dpi = 80
#fig = plt.figure(dpi = dpi, figsize = (812 / dpi, 384 / dpi) )
mpl.rcParams.update({'font.size': 10})

#plt.axis([0, 10, -1.5, 1.5])

#plt.title('graphics')
#plt.xlabel('x')
#plt.ylabel('F(x)')


#plt.plot(xs, sin_vals, color = 'blue', linestyle = 'solid',
 #        label = 'sin(x1)')

#table = pd.read_excel('C:\Users\User\Desktop\Practikaset\data1.xlsx')
#x = table.values[:, 0]
#y = table.values[:, 1]

#plt.figure(figsize=(15, 7))
#plt.plot(x, y)



# Train network
#train_neural_network(xx)

def model(X, w):
    return tf.multiply(X, w)  



tk.mainloop()
