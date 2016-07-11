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
import cPickle as pickle

#dirs = [42,38,39,40,41]
dirs = [80,]

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )
#print ddict

show_avg_force = False
fft = False
calibrate = True
init_data = [0., 0., 20.]

load_charge_cal = True
files = np.arange(0,40,1)
maxfiles = 1000

#################

if not load_charge_cal:
    cal = [['/data/20160627/bead1/chargelp_withap_2nd_cal2'], 'Cal', 20]

    cal_dir_obj = cu.Data_dir(cal[0], [0,0,cal[2]], cal[1])
    cal_dir_obj.load_dir(cu.simple_loader)
    cal_dir_obj.build_step_cal_vec()
    cal_dir_obj.step_cal()
    cal_dir_obj.save_step_cal('./calibrations/step_cal_20160701.p')

    for fobj in cal_dir_obj.fobjs:
        fobj.close_dat()

    step_calibration = cal_dir_obj.charge_step_calibration

#################

def proc_dir(d):
    dv = ddict[d]

    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)

time_dict = {}
for obj in dir_objs:
    for fobj in obj.fobjs:
        time = fobj.Time
        if time not in time_dict:
            time_dict[time] = []
            time_dict[time].append(fobj.fname)
        else:
            time_dict[time].append(fobj.fname)


times = time_dict.keys()

colors_yeay = bu.get_color_map( len(times) )

for i, time in enumerate(times):
    if i not in files:
        continue

    newobj = cu.Data_dir(0, init_data, time)
    newobj.files = time_dict[time]
    newobj.load_dir(cu.diag_loader, maxfiles=maxfiles)

    newobj.load_H("./trans_funcs/Hout_20160630.p")
    
    if load_charge_cal:
        newobj.load_step_cal('./calibrations/step_cal_20160701.p')
    else:
        newobj.charge_step_calibration = step_calibration

    newobj.diagonalize_files()#simpleDCmat=True)

    newobj.get_conv_facs()

    col = colors_yeay[i]
    if calibrate:
        cal_facs = newobj.conv_facs
    else:
        cal_facs = [1.,1.,1.]
    newobj.get_avg_force_v_pos(axis = 1, bin_size = 4)
    newobj.get_avg_diag_force_v_pos(axis = 1, bin_size = 4)

    keys = newobj.avg_force_v_pos.keys()
    for key in keys:
        offset = 0
        #offset = -1.0 * obj.avg_force_v_pos[key][1][-1]
        lab = newobj.label
        plt.figure(1)
        plt.errorbar(newobj.avg_force_v_pos[key][0], (newobj.avg_force_v_pos[key][1] + offset) * cal_facs[1] * 1e15, newobj.avg_force_v_pos[key][2] * cal_facs[1] * 1e15, label = lab, fmt='.-', ms=10)#, color = col)
        plt.figure(2)
        plt.errorbar(newobj.avg_diag_force_v_pos[key][0], (newobj.avg_diag_force_v_pos[key][1] + offset) * 1e15, newobj.avg_diag_force_v_pos[key][2] * 1e15, label = lab, fmt='.-', ms=10)#, color = col)

for fig in [1,2]:
    plt.figure(fig)
    plt.xlabel('Distance from Cantilever [um]')
    plt.ylabel('Force [fN]')
    plt.legend(loc=0, numpoints=1)


plt.show()
