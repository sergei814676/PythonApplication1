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
import os.path

import metods
import class_network_files
import working_class


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
     #  self.ax = self.fig.add_subplot(3,1,1) # add an Axes to the figure
      # PythonApplication1.root.destroy()
      # for item in self.bar2.get_tk_widget().find_all():
      #  self.bar2.get_tk_widget().delete(item)

     #  self.ax.plot(history.history['loss'], label='train')
      # self.ax.plot(history.history['val_loss'], label='test')


     # np.transpose

       x1=(np.arange(float(PythonApplication1.message_entry_from_x.get()),  float(PythonApplication1.message_entry_to_x.get()), float(PythonApplication1.message_entry_interval_x.get())))
       y1 =(np.arange(float(PythonApplication1.message_entry_from_y.get()),  float(PythonApplication1.message_entry_to_y.get()), float(PythonApplication1.message_entry_interval_y.get())))
       
       self.ax = self.fig.add_subplot(1,1,1,projection='3d')
       
       self.ax.scatter(x_p,y_p,z_p,color='r')

       
       xgrid, ygrid = np.meshgrid(x1, y1)
       XYpairs = np.dstack([xgrid, ygrid]).reshape(-1, 2)
       #zgrid= PythonApplication1.Main_class.Model_main.predict([x,y])
       q1,q2 = np.shape(xgrid)
       
       XYpairs2=PythonApplication1.Main_class.Model_main.predict(XYpairs)
       XYpairs2= np.reshape(XYpairs2,(q1,q2))


       self.ax.plot_surface(xgrid, ygrid ,XYpairs2, lw=0, cmap='spring')       
       
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
   
    def draw_plot_2d (self,model):
      # self.fig.close()
      
      # plt.close(self.fig)
       self.fig.clf()
       #self.fig = plt.Figure() # create a figure object
       #self.ax = self.fig.add_subplot(111)
       
       xx=np.arange(float(message_entry_from_x.get()),  float(message_entry_to_x.get()), float(message_entry_interval_x.get()))
     #  self.fig = plt.figure(figsize=(10, 10), dpi=77)
       self.ax = self.fig.add_subplot(211) # add an Axes to the figure
      # PythonApplication1.root.destroy()
      # for item in self.bar2.get_tk_widget().find_all():
      #  self.bar2.get_tk_widget().delete(item)

#       self.ax.plot(history.history['loss'], label='train')
#       self.ax.plot(history.history['val_loss'], label='test')
#       self.ax = self.fig.add_subplot(212)
       
       #x=np.arange(min(x[:,0]), max(x[:,0]), 0.1) # xx=np.arange(min(x[:,0]), max(x[:,0]), 0.1)
       self.ax.plot(xx, model.predict(xx),c='#2ca02c')

       if (os.path.exists(PythonApplication1.message.get())):
        x, y = metods.file_acceptance()
        self.ax.scatter(x[:,0], y,c='#ff7f0e') #self.ax.scatter(x[:,0], y,c='#ff7f0e')
        MSE_Deviation=mean_squared_error(y,model.predict(x[:,0]))
        message_entry_Deviation.delete(0,END)
        message_entry_Deviation.insert(0, MSE_Deviation)


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
       
def rounded_rect(canvas, x, y, w, h, c): #функция для скругления углов
    canvas.create_arc(x,   y,   x+2*c,   y+2*c,   start= 90, extent=90, style="arc")
    canvas.create_arc(x+w-2*c, y+h-2*c, x+w, y+h, start=270, extent=90, style="arc")
    canvas.create_arc(x+w-2*c, y,   x+w, y+2*c,   start=  0, extent=90, style="arc")
    canvas.create_arc(x,   y+h-2*c, x+2*c,   y+h, start=180, extent=90, style="arc")
    #canvas.create_line(x+c, y,   x+w-c, y    )
   # canvas.create_line(x+c, y+h, x+w-c, y+h  )
   # canvas.create_line(x,   y+c, x,     y+h-c)
   # canvas.create_line(x+w, y+c, x+w,   y+h-c)


def choose_file():  # считывание файла
        filetypes =[('Файл необходимый', '*.xlsx *.csv'), ('Все файлы', '*')]
        dlg = fd.Open(filetypes = filetypes)
        fl = dlg.show()
        if fl != '':
            message_entry.delete(0, END)
            message_entry.insert(0, fl)
        if (os.path.exists(PythonApplication1.message_entry.get())):
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
         




network_files_clas= class_network_files.class_network_files()
existence_working_class=0
Main_class=working_class.working_class()
Main_class.Model_main=Main_class.NewOneNetwork()

def new_files():# команды для меню верхнего
    PythonApplication1.existence_working_class=1
    network_files_clas.new_network_files() 


def save_files():# команды для меню верхнего
    if (existence_working_class==0):
        messagebox.showinfo('Информация', 'Нечего сохранять') 
    if (existence_working_class==1):
        network_files_clas.save_network_files(message_entry_Name_network.get(),Main_class.Model_main) 

    
   # messagebox.showinfo('О программе', 'Сохранить?') 

def load_files():# команды для меню верхнего
    network_files_clas.load_network_files() 
    PythonApplication1.existence_working_class=1 


def about_the_program():
    messagebox.showinfo('О программе','Сделано НСН для НСН' ) 





        #if filename:
          #  print(filename)
       # message_entry.insert(0, fl)


root = Tk()
root.title("Аппроксимация машинным обучением")
root.geometry("1050x500")
root.configure(bg='#49A')

#canvas = tk.Canvas(root)
#canvas.pack()
#rounded_rect(canvas, 20, 20, 60, 40, 10)
#root.mainloop()


menu = Menu(root)  
new_item_info = Menu(menu) 

new_item_files = Menu(menu)

new_item_files.add_command(label='Новая сеть' , command=new_files)
new_item_files.add_command(label='Сохранить сеть' , command=save_files)
new_item_files.add_command(label='Загрузить сеть' , command=load_files)
menu.add_cascade(label='Файл', menu=new_item_files)


new_item_info.add_command(label='О Создателе' , command=about_the_program)  
menu.add_cascade(label='О программе', menu=new_item_info)  

root.config(menu=menu)  
name_colum=['a','b','c']
message_from_x = StringVar()
message_to_x = StringVar()
message_interval_x=StringVar()
message_from_y=StringVar()
message_to_y=StringVar()
message_interval_y=StringVar()
message_number_of_epochs=StringVar()
message_test_percentage=StringVar()
message_Deviation=StringVar()

#label13 = Label(text="Топология нейросети:",bg='#49A')
#label13.place(relx=0.04, rely=0.23)

label12 = Label(text="Визуализация:",bg='#49A')
label12.place(relx=0.04, rely=0.33)

label2 = Label(text="Х   От",bg='#49A')
label2.place(relx=0.04, rely=0.48)
message = StringVar()

label3 = Label(text="До",bg='#49A')
label3.place(relx=0.14, rely=0.48)

label4 = Label(text="Интервал:",bg='#49A')
label4.place(relx=0.04, rely=0.55)


label5 = Label(text="Y   От",bg='#49A')
label5.place(relx=0.04, rely=0.60)
message = StringVar()

label6 = Label(text="До", bg='#49A')
label6.place(relx=0.14, rely=0.60)

label7 = Label(text="Интервал:", bg='#49A')
label7.place(relx=0.04, rely=0.65)

label8 = Label(text="СКО=", bg='#49A')
label8.place(relx=0.04, rely=0.75)

label8 = Label(text="Имя сети:", bg='#49A')
label8.place(relx=0.05, rely=0.02)



label9 = Label(text="Кол. эпох",bg='#49A')
label9.place(relx=0.04, rely=0.80)
message = StringVar()

label10 = Label(text="% тестовой", bg='#49A')
label10.place(relx=0.144, rely=0.80)


message_entry_number_of_epochs = Entry(textvariable=message_number_of_epochs, width=7)
message_entry_number_of_epochs.place(relx=0.1, rely=0.8)
message_entry_number_of_epochs.insert(0,"1000")

message_entry_test_percentage = Entry(textvariable=message_test_percentage, width=7)
message_entry_test_percentage.place(relx=0.211, rely=0.8)
message_entry_test_percentage.insert(0,"15")

message_entry_Name_network = Entry(textvariable="Name1", width=20)
message_entry_Name_network.place(relx=0.13, rely=0.02)

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

def pzdz():
    PythonApplication1.Main_class.fit_models()

def nah():
    PythonApplication1.Main_class.Graph_clas()
 
message_entry = Entry(textvariable=message)
message_entry.place(relx=.1, rely=.1, anchor="c")

btn_file = Button(text="Выбрать файл", command=choose_file)
btn_file.place(relx=.2, rely=.1, anchor="c") 

message_button = Button ( text='Обучение', command = pzdz) #metods.choiceCombobox
message_button.place(relx=.1, rely=.2, anchor="c")


message_button = Button(text="Результат", command=nah) #metods.choiceCombobox
message_button.place(relx=.2, rely=.2, anchor="c")

comboExample1 = ttk.Combobox(root, 
                            values=["Ориджин",
                                    "Matplotlib",
                                    "CSV", 
                                    ],
                            state="readonly")


#comboExample = ttk.Combobox(root, 
                        #    values=["RNN",
                                    #"PodborZnacheny",
                               #     "NewOneNetwork", 
                                   # "Polinel",
                                   # "probanerset",
                                   # "Lineq",
                            #        "TandemNN",
                            #        "NewFourNetwork",
                              #      "NewFiveNetwork"],
                          #  state="readonly")
#comboExample.place(relx=.1, rely=.3, anchor="c")
comboExample1.place(relx=.1, rely=.4, anchor="c")
#comboExample.current(1)
comboExample1.current(1)
#comboExample1['state'] = 'disabled'
#comboExample.pack_forget()

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



    

    
