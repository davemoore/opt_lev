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

#dirs = [29,30,31,32]
dirs = [55,]#[45,54,55,56]  # 45 is the old one without ap
cal = 5.0e-14

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )

load_from_file = False
load_charge_cal = True
maxfiles = 1000

#################################

if not load_charge_cal:
    cal = [['/data/20160628/bead1/chargelp_cal3'], 'Cal', 20, 1e-13]

    cal_dir_obj = cu.Data_dir(cal[0], [0,0,cal[2]], cal[1])
    cal_dir_obj.load_dir(cu.simple_loader)
    cal_dir_obj.build_step_cal_vec()
    cal_dir_obj.step_cal()
    cal_dir_obj.save_step_cal('./calibrations/step_cal_20160628.p')

    for fobj in cal_dir_obj.fobjs:
        fobj.close_dat()

    step_calibration = cal_dir_obj.charge_step_calibration

#################################




def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.H_loader, maxfiles = maxfiles)

    dir_obj.build_uncalibrated_H(average_first=True)
    
    if load_charge_cal:
        dir_obj.load_step_cal('./calibrations/step_cal_20160701.p')
    else:
        dir_obj.charge_step_calibration = step_calibration

    dir_obj.calibrate_H()
    dir_obj.get_conv_facs()
    
    if '06-28' in dir_obj.label:
        dir_obj.thermal_cal_file_path = '/data/20160627/bead1/1_5mbar_nocool.h5'
    elif '06-30' in dir_obj.label:
        dir_obj.thermal_cal_file_path = '/data/20160627/bead1/1_5mbar_nocool_withap.h5'

    dir_obj.thermal_calibration()

    dir_obj.build_Hfuncs(fpeaks=[245, 255, 50], weight_peak=False, weight_above_thresh=True,\
                         plot_fits=True, weight_phase=True)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)






counter = 0
for obj in dir_objs:
    if obj == dir_objs[-1]:
        obj.thermal_cal_fobj.plt_thermal_fit()



for obj in dir_objs:
    if obj != dir_objs[-1]:
        obj.plot_H(phase=True, show=False, label=True, show_zDC=True, \
                   inv=False, lim=False)
        #obj.plot_H(phase=False, label=False, show=False, noise=True)
        continue

    obj.plot_H(phase=True, label=True, show=True, show_zDC=True, \
               inv=False, lim=False)
    #obj.plot_H(phase=False, label=False, show=True, noise=True)
    obj.save_H("./trans_funcs/Hout_20160630.p")
