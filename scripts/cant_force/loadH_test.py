import cant_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import Tkinter
import tkFileDialog
import os, sys
from scipy.optimize import curve_fit
import bead_util as bu
from scipy.optimize import minimize_scalar as minimize 

dirs = [13,14,15,16]
cal = 5.0e-14

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )

load_from_file = False

new_obj = cu.Data_dir('shit_path', [0,0,0], "wheeeee")

new_obj.load_H("optphase2_Hout.p")
new_obj.plot_H(phase=True, label=True)
