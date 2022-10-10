#coding=<utf8>

import tkinter.filedialog as fd
import keras
import PythonApplication1
import os.path
from tkinter import messagebox

import tkinter as tk
from tkinter import *

class class_network_files(object):

    def __init__(self):
        """Constructor"""
  
        #ax = self.fig.add_subplot(111)
        #self.fig.clear()

    def new_network_files (self):
       

       #self.askdirectory =[('Folder', '*'), ('All files', '*')]
     # self.dlg = fd.askdirectory(filetypes = self.askdirectory)
       self.fl = fd.asksaveasfilename(
       filetypes=(("H5 files", "*.h5"),
                   ("All files", "*.*")))
      
     #  self.fl = fd.askdirectory(title="Open Folder" )

     #  self.fl = self.dlg.show()
       if self.fl != '':
           PythonApplication1.message_entry_Name_network.delete(0, END)
           PythonApplication1.message_entry_Name_network.insert(0, self.fl+'.h5')
       PythonApplication1.Main_class.Model_main = PythonApplication1.Main_class.NewOneNetwork()
       PythonApplication1.Main_class.train=0


    def save_network_files (self, file_path, model_network):
        
        
        #PythonApplication1.Main_class.save(file_path)
        model_network.save(file_path,  overwrite=True, include_optimizer=True,save_format='h5')
       
       

    def load_network_files (self):
       self.fl = fd.askopenfilename(
       filetypes=(("H5 files", "*.h5"),
                   ("All files", "*.*")))
      
     #  self.fl = fd.askdirectory(title="Open Folder" )

     #  self.fl = self.dlg.show()
     
     #  self.filetypes =[('Folder', '*'), ('All files', '*')]
     # self.dlg = fd.Open(filetypes = self.filetypes)
     #  self.fl = self.dlg.show()
      # if self.fl != '':
       PythonApplication1.message_entry_Name_network.delete(0, END)
       PythonApplication1.message_entry_Name_network.insert(0, self.fl)
       if (os.path.exists(PythonApplication1.message_entry_Name_network.get())):
           PythonApplication1.Main_class.Model_main= keras.models.load_model(self.fl)
       else:
           messagebox.showinfo('Информация', 'Файл не выбран!')
       #PythonApplication1.Main_class.Model_main= keras.models.load_model(self.fl)

    def delete_network_files (self, file_path, model_network):
       
       
       self.bar2.draw()

    pass




