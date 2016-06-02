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

dirs = [10,]
cal = 5.0e-14

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )

load_from_file = False


def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-2]])
    dir_obj.load_dir(cu.H_loader)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)

#cal_dir = cu.Data_dir(ddict[11][0], [0,0,ddict[11][-2]])
#cal_dir.load_dir(cu.H_loader)

counter = 0
for obj in dir_objs:

    #obj.step_cal(cal_dir)
    #print obj.charge_step_calibration

    obj.plot_H(phase=True)
