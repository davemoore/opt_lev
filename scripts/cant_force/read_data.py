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
dirs = [61, 62, 63, 64] # [47,48,49,50,51,52,53]

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )
#print ddict

load_from_file = False
show_each_file = False
show_avg_force = False
fft = False
calibrate = True

load_charge_cal = True
maxfiles = 1000

#################

if not load_charge_cal:
    cal = [['/data/20160627/bead1/chargelp_cal3'], 'Cal', 20]

    cal_dir_obj = cu.Data_dir(cal[0], [0,0,cal[2]], cal[1])
    cal_dir_obj.load_dir(cu.simple_loader)
    cal_dir_obj.build_step_cal_vec()
    cal_dir_obj.step_cal()
    cal_dir_obj.save_step_cal('./calibrations/step_cal_20160628.p')

    for fobj in cal_dir_obj.fobjs:
        fobj.close_dat()

    step_calibration = cal_dir_obj.charge_step_calibration


#################

def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.diag_loader, maxfiles = maxfiles)

    dir_obj.load_H("./trans_funcs/Hout_20160630.p")
    
    if load_charge_cal:
        dir_obj.load_step_cal('./calibrations/step_cal_20160701.p')
    else:
        dir_obj.charge_step_calibration = step_calibration

    dir_obj.diagonalize_files()

    dir_obj.get_conv_facs()

    #dir_obj.plot_H(cal=True)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)

thermal_cal_file_path = '/data/20160627/bead1/1_5mbar_zcool.h5'


colors_yeay = bu.get_color_map( len(dir_objs)+1 )
for i, obj in enumerate(dir_objs):
    col = colors_yeay[i]
    if calibrate:
        cal_facs = obj.conv_facs
    else:
        cal_facs = [1.,1.,1.]
    obj.get_avg_force_v_pos(axis = 1, bin_size = 4)
    obj.get_avg_diag_force_v_pos(axis = 1, bin_size = 4)

    keys = obj.avg_force_v_pos.keys()
    for key in keys:
        offset = 0
        #offset = -1.0 * obj.avg_force_v_pos[key][1][-1]
        lab = obj.label
        plt.figure(1)
        plt.errorbar(obj.avg_force_v_pos[key][0], (obj.avg_force_v_pos[key][1] + offset) * cal_facs[1] * 1e15, obj.avg_force_v_pos[key][2] * cal_facs[1] * 1e15, label = lab, fmt='.-', ms=10)#, color = col)
        plt.figure(2)
        plt.errorbar(obj.avg_diag_force_v_pos[key][0], (obj.avg_diag_force_v_pos[key][1] + offset) * 1e15, obj.avg_diag_force_v_pos[key][2] * 1e15, label = lab, fmt='.-', ms=10)#, color = col)

for fig in [1,2]:
    plt.figure(fig)
    plt.xlabel('Distance from Cantilever [um]')
    plt.ylabel('Force [fN]')
    plt.legend(loc=0, numpoints=1)


plt.show()
