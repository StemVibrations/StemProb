import json
import os



import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


import tkinter
root = tkinter.Tk()
print("Tcl library path:", root.tk.globalgetvar('tcl_library'))
try:
    print("Tk library path:", root.tk.globalgetvar('tk_library'))
except:
    print("Tk library path not set directly")
root.destroy()

