import matplotlib
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import math, random
import pandas as pd
from PIL import ImageTk
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.metrics import mean_squared_error



# This is the button callback function
# This must be visible to the button, so we must define it before the button widget!



class Graph_in_indow():
    """docstring"""
 
    def __init__(self):
        """Constructor"""
        self.fig = plt.figure(figsize=(10, 10), dpi=78)
       
        self.ax = plt.axes(projection='3d')
        #self.ax = self.fig.add_subplot(1,1,1)
        #fig.add_subplot(111)
        self.bar2 = FigureCanvasTkAgg(self.fig, PythonApplication1.root)
        self.bar2.get_tk_widget().pack(expand=1, anchor=SE)
        #ax = self.fig.add_subplot(111)
        #self.fig.clear()


    def draw_plot_3d (self,x,y,z,x_p,y_p,z_p,history,model):
       
        # self.fig.close()
      
      # plt.close(self.fig)
       self.fig.clf()
       #self.fig = plt.Figure() # create a figure object
       #self.ax = self.fig.add_subplot(111)

     #  self.fig = plt.figure(figsize=(10, 10), dpi=77)
       self.ax = self.fig.add_subplot(3,1,1) # add an Axes to the figure
      # PythonApplication1.root.destroy()
      # for item in self.bar2.get_tk_widget().find_all():
      #  self.bar2.get_tk_widget().delete(item)

       self.ax.plot(history.history['loss'], label='train')
       self.ax.plot(history.history['val_loss'], label='test')
       self.ax = self.fig.add_subplot(3,1,(2,3),projection='3d')
       
       self.ax.scatter(x_p,y_p,z_p,color='r')
       self.ax.plot_surface(x, y ,  z, lw=0, cmap='spring')       
       
       xx=np.arange(min(x[:,0]), max(x[:,0]), 0.1)


       xy_p=np.ones((len(x_p),2))
       xy_p[:,0]=x_p
       xy_p[:,1]=y_p

       MSE_Deviation=mean_squared_error(z_p,model.predict(xy_p))



       message_entry_Deviation.delete(0,END)
       message_entry_Deviation.insert(0, MSE_Deviation)
    
       self.bar2.draw()
       

        #self.ax = plt.axes(projection='3d')

        #self.bar2 = FigureCanvasTkAgg(self.fig, PythonApplication1.root)
        #self.bar2.get_tk_widget().pack(expand=1, anchor=SE, fill=X)
   
    def draw_plot_2d (self,model,x,y,history):
      # self.fig.close()
      
      # plt.close(self.fig)
       self.fig.clf()
       #self.fig = plt.Figure() # create a figure object
       #self.ax = self.fig.add_subplot(111)

     #  self.fig = plt.figure(figsize=(10, 10), dpi=77)
       self.ax = self.fig.add_subplot(211) # add an Axes to the figure
      # PythonApplication1.root.destroy()
      # for item in self.bar2.get_tk_widget().find_all():
      #  self.bar2.get_tk_widget().delete(item)

       self.ax.plot(history.history['loss'], label='train')
       self.ax.plot(history.history['val_loss'], label='test')
       self.ax = self.fig.add_subplot(212)
       self.ax.scatter(x[:,0], y,c='#ff7f0e')
       xx=np.arange(min(x[:,0]), max(x[:,0]), 0.1)
       self.ax.plot(xx, model.predict(xx),c='#2ca02c')


       self.bar2.draw()



       #self.fig = plt.figure(figsize=(10, 10), dpi=77)
       #self.fig, self.ax = plt.subplots()
       #ax = plt.Axes(self.fig)
      # self.ax = plt.axes(fig)
        #self.ax = self.fig.add_subplot(1,1,1)
        #fig.add_subplot(111)
       #self.bar2 = FigureCanvasTkAgg(self.fig, PythonApplication1.root)
       #self.bar2.get_tk_widget().pack(expand=1, anchor=SE)
       #self.ax.clear()
       #self.ax = plt.axes(projection='3d')
      # self.ax = plt.scatter(x[:,0], y)
      # self.fig.ax('off')
      # self.ax = plt.axes()
      # self.ax = self.fig.add_subplot(211)
      # self.ax.scatter(x[:,0], y)
      # xx=np.arange(min(x[:,0]), max(x[:,0]), 0.1)
      # self.ax.scatter(x[:,0], model.predict(x), s=1)
     #  self.ax = self.fig.add_subplot(212)
      # self.ax.plot(history.history['loss'], label='train')
      # self.ax.plot(history.history['val_loss'], label='test')
      # self.ax.legend()
      # self.bar2 = FigureCanvasTkAgg(self.fig, PythonApplication1.root)
      # self.bar2.get_tk_widget().pack(expand=1, anchor=SE, fill=X)
      # self.bar2.draw()
      # self.bar2.show()

        #self.ax.fig.clear()
       # self.ax.clear()
        #self.ax = plt.axes(projection='3d')
    #    self.ax.scatter(x_p,y_p,s=15,color='r')
     #   self.ax.scatter(x, y)
      #  self.bar2.draw()
        #self.bar2 = FigureCanvasTkAgg(self.fig, PythonApplication1.root)
        #self.bar2.get_tk_widget().pack(expand=1, anchor=SE, fill=X)
       



def choose_file():
        filetypes =[('???????? ??????????????????????', '*.xlsx *.csv'), ('?????? ??????????', '*')]
        dlg = fd.Open(filetypes = filetypes)
        fl = dlg.show()
        if fl != '':
            message_entry.delete(0, END)
            message_entry.insert(0, fl)
        
        x,y=metods.file_acceptance()
        x_rows, x_cols = x.shape
        if (x_cols==2):
            message_entry_interval_x.delete(0)
            message_entry_from_x.delete(0)
            message_entry_to_x.delete(0)
            message_entry_interval_y.delete(0)
            message_entry_from_y.delete(0)
            message_entry_to_y.delete(0)

           
            message_entry_from_x.insert(0, min(x[:,0]))
            message_entry_to_x.insert(0, max(x[:,0]))
            
            message_entry_from_y.insert(0, min(x[:,1]))
            message_entry_to_y.insert(0, max(x[:,1]))
            if (max(x[:,0]-min(x[:,0])>10)):
                  message_entry_interval_x.delete(0,END)
                  message_entry_interval_x.insert(0, 1)
            if (max(x[:,1]-min(x[:,1])>10)):
                  message_entry_interval_y.delete(0,END)
                  message_entry_interval_y.insert(0, 1)


            if (((max(x[:,0])-min(x[:,0]))>1) and ((max(x[:,0])-min(x[:,0]))<=10)):
                  message_entry_interval_x.delete(0,END)
                  message_entry_interval_x.insert(0, 0.1)
            if (((max(x[:,1])-min(x[:,1]))>1) and ((max(x[:,1])-min(x[:,1]))<=10)):
                  message_entry_interval_y.delete(0,END)
                  message_entry_interval_y.insert(0, 0.1)

            if (((max(x[:,0])-min(x[:,0]))>0.1) and ((max(x[:,0])-min(x[:,0]))<=10)):
                  message_entry_interval_x.delete(0,END)
                  message_entry_interval_x.insert(0, 0.01)
            if (((max(x[:,1])-min(x[:,1]))>0.1) and ((max(x[:,1])-min(x[:,1]))<=10)):
                  message_entry_interval_y.delete(0,END)
                  message_entry_interval_y.insert(0, 0.01)



        if (x_cols==1):
            message_entry_interval_x.delete(0)
            message_entry_from_x.delete(0)
            message_entry_to_x.delete(0)


           
            message_entry_from_x.insert(0, min(x[:,0]))
            message_entry_to_x.insert(0, max(x[:,0]))
  
            if (max(x)-min(x)>10):
                  message_entry_interval_x.delete(0,END)
                  message_entry_interval_x.insert(0, 1)
          


            if ((max(x)-min(x))>1) and (max(x)-min(x)<=10):
                  message_entry_interval_x.delete(0,END)
                  message_entry_interval_x.insert(0, 0.1)
          

            if ((max(x)-min(x))>0.1) and ((max(x)-min(x))<=1):
                  message_entry_interval_x.delete(0,END)
                  message_entry_interval_x.insert(0, 0.01)
         











def about_the_program():
    messagebox.showinfo('?? ??????????????????', '?????????????? ?????? ?????? ??????') 





        #if filename:
          #  print(filename)
       # message_entry.insert(0, fl)


root = Tk()
root.title("?????????????????????????? ???????????????? ??????????????????")
root.geometry("1050x500")




menu = Menu(root)  
new_item = Menu(menu)  
new_item.add_command(label='?? ??????????????????' , command=about_the_program)  
menu.add_cascade(label='?? ??????????????????', menu=new_item)  
root.config(menu=menu)  
name_colum=['a','b','c']
message_from_x = StringVar()
message_to_x = StringVar()
message_interval_x=StringVar()
message_from_y=StringVar()
message_to_y=StringVar()
message_interval_y=StringVar()

message_Deviation=StringVar()



label2 = Label(text="From")
label2.place(relx=0.04, rely=0.48)
message = StringVar()

label3 = Label(text="To")
label3.place(relx=0.14, rely=0.48)

label4 = Label(text="With an interval")
label4.place(relx=0.04, rely=0.55)


label5 = Label(text="From")
label5.place(relx=0.04, rely=0.60)
message = StringVar()

label6 = Label(text="To")
label6.place(relx=0.14, rely=0.60)

label7 = Label(text="With an interval")
label7.place(relx=0.04, rely=0.65)

label8 = Label(text="Deviation")
label8.place(relx=0.04, rely=0.75)

message_entry_Deviation = Entry(textvariable=message_Deviation, width=7)
message_entry_Deviation.place(relx=0.13, rely=0.75)


message_entry_interval_y = Entry(textvariable=message_interval_y, width=7)
message_entry_interval_y.place(relx=0.13, rely=0.65)

message_entry_from_y = Entry(textvariable=message_from_y, width=7)
message_entry_from_y.place(relx=.1, rely=0.62, anchor="c")

message_entry_to_y = Entry(textvariable=message_to_y, width=7)
message_entry_to_y.place(relx=.2, rely=0.62, anchor="c")


message_entry_interval_x = Entry(textvariable=message_interval_x, width=7)
message_entry_interval_x.place(relx=0.13, rely=0.55)

message_entry_from_x = Entry(textvariable=message_from_x, width=7)
message_entry_from_x.place(relx=.1, rely=.5, anchor="c")

message_entry_to_x = Entry(textvariable=message_to_x, width=7)
message_entry_to_x.place(relx=.2, rely=.5, anchor="c")

#image = ImageTk.PhotoImage(file="strelka levo.png")

#message_button = Button(image=image,command=lambda: print('click'))
#message_button.place(relx=1.2, rely=.5, anchor="c")


 
message_entry = Entry(textvariable=message)
message_entry.place(relx=.1, rely=.1, anchor="c")

btn_file = Button(text="?????????????? ????????", command=choose_file)
btn_file.place(relx=.2, rely=.1, anchor="c") 

message_button = Button(text="????????????", command=metods.choiceCombobox)
message_button.place(relx=.1, rely=.2, anchor="c")


comboExample1 = ttk.Combobox(root, 
                            values=["??????????????",
                                    "????????????????",
                                    "CSV", 
                                    ],
                            state="readonly")


comboExample = ttk.Combobox(root, 
                            values=["RNN",
                                    #"PodborZnacheny",
                                    "NewOneNetwork", 
                                   # "Polinel",
                                   # "probanerset",
                                   # "Lineq",
                                    "TandemNN",
                                    "NewFourNetwork",
                                    "NewFiveNetwork"],
                            state="readonly")
comboExample.place(relx=.1, rely=.3, anchor="c")
comboExample1.place(relx=.1, rely=.4, anchor="c")
comboExample.current(1)
comboExample1.current(1)

tf.disable_v2_behavior()
grach=Graph_in_indow()
#df1 = pd.DataFrame(data1,columns=['Country','GDP_Per_Capita'])
#figure1 = plt.Figure(figsize=(8,5), dpi=75)
#ax1 = figure1.add_subplot(111)
#bar1 = FigureCanvasTkAgg(figure1, root)
#bar1.get_tk_widget().pack(expand=1, anchor=S, fill=X)

#df1 = df1[['Country','GDP_Per_Capita']].groupby('Country').sum()

#df1.plot(kind='bar', legend=True, ax=ax1)
#ax1.set_title('Country Vs. GDP Per Capita')
#figure1.clf()


dpi = 80
#fig = plt.figure(dpi = dpi, figsize = (812 / dpi, 384 / dpi) )
mpl.rcParams.update({'font.size': 10})
ttt=0
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



    

    
