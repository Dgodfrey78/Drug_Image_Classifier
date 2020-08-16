from tkinter import *
from tkinter import ttk
import sys, os



from ImageClassify import isDrug
from ImageClass import whichDrug

import sys, os



window = Tk()

window.title("Don't Do Drugs")

window.geometry('400x200')



w=Label(window, text="Is there drugs in your image?")

x= Label(window, text="What drug is in your image?")

lbl = Label(window, text="Image File Name")

y= Label(window, text="Image File Name")

w.grid(column=0,row=0)

lbl.grid(column=0, row=1)

txt = Entry(window,width=10)

txt2 = Entry(window,width=10)

txt.grid(column=1, row=1)

x.grid(column=0,row=2)

txt2.grid(column=1,row=3)

y.grid(column=0,row=3)



def clicked():

    res = txt.get()
    res2 = txt2.get()
    
    isDrug(res)
    whichDrug(res2)

btn = Button(window, text="Analyze", command=clicked)

btn.grid(column=1, row=4)

window.mainloop()
