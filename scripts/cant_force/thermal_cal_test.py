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

dirs = [23,27]
cal = 5.0e-14

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )

load_from_file = False

'''
def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-2]], dv[1])
    dir_obj.load_dir(cu.H_loader, maxfiles=30)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)
'''


cal_obj = cu.Data_file()
cal_obj.load("/data/20160601/bead1/1_5mbar_zcool_aperatureadjust.h5", [0,0,20])

cal_obj.thermal_calibration()
#cal_obj.plt_thermal_fit()

norm_rat = cal_obj.cal_fac()
norm_rat2 = cal_obj.cal_fac(axis=1)
norm_rat3 = cal_obj.cal_fac(axis=2)

plt.figure()
plt.loglog(cal_obj.psd_freqs, np.sqrt(cal_obj.psds[0] * norm_rat))
plt.loglog(cal_obj.psd_freqs, np.sqrt(cal_obj.psds[1] * norm_rat2))
plt.show()

print [np.sqrt(norm_rat), np.sqrt(norm_rat2), np.sqrt(norm_rat3)]
