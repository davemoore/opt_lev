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

dirs = [19,20,21,22]
cal = 5.0e-14

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )

def proc_dir(d):
    dv = ddict[d]

    blind_force_ind = int(dv[1][-1])

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-2]], dv[1])
    dir_obj.load_dir(cu.ft_loader)
    dir_obj.load_H("optphase2_Hout.p")

    diag_drive = dir_obj.diagonalize_ave_pos()

    plt.figure()
    for i in [0,1,2]:
        plt.subplot(3,1,1+i)
        plt.plot(diag_drive[i], label=dv[1])
    #plt.show()

    return dir_obj

dir_objs = map(proc_dir, dirs)
plt.legend()
plt.show()
