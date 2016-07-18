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
import time

dirs = [156,157,158,159,160,161,162]

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )
#print ddict

fft = False
calibrate = True

respdir = 'Y'
resp_axis = 1
cant_axis = 2
bin_size = 5

load_charge_cal = True
maxfiles = 1000

subtract_background = True

fig_title = 'Force vs. Cantilever Position: Various Pressures'

tf_path = './trans_funcs/Hout_20160715.p'
step_cal_path = './calibrations/step_cal_20160715.p'


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

    dir_obj.load_H(tf_path)
    
    if load_charge_cal:
        dir_obj.load_step_cal(step_cal_path)
        #dir_obj.load_step_cal('./calibrations/step_cal_TEST.p')
    else:
        dir_obj.charge_step_calibration = step_calibration

    dir_obj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=200., # plot_Happ=True, \
                             build_conv_facs=True, drive_freq=18.)

    return dir_obj

dir_objs = map(proc_dir, dirs)

thermal_cal_file_path = '/data/20160715/bead1/1_5mbar_zcool_final2.h5'


colors_yeay = bu.get_color_map( len(dir_objs) )
f, axarr = plt.subplots(3,2,sharey='all',sharex='all',figsize=(10,12),dpi=100)

if subtract_background:
    f2, axarr2 = plt.subplots(3,2,sharey='all',sharex='all',figsize=(10,12),dpi=100)
    sub_dat = [[],[],[]]
    sub_dat_d = [[],[],[]]

for i, obj in enumerate(dir_objs):
    col = colors_yeay[i]
    if calibrate:
        cal_facs = obj.conv_facs
    else:
        cal_facs = [1.,1.,1.]

    obj.get_avg_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)

    obj.get_avg_diag_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)

    keys = obj.avg_force_v_pos.keys()
    for key in keys:
        lab = '%0.4f+%0.5f' %(obj.avg_pressure[2], obj.var_pressure[2])

        for resp_axis in [0,1,2]:
            if resp_axis == 1:
                offset = -1.0 * obj.avg_force_v_pos[key][resp_axis,0][1][-1]
                offset_d = -1.0 * obj.avg_diag_force_v_pos[key][resp_axis,0][1][-1]
            else:
                offset = 0
                offset_d = 0
            xdat = obj.avg_force_v_pos[key][resp_axis,0][0]
            ydat = (obj.avg_force_v_pos[key][resp_axis,0][1] + offset) * cal_facs[resp_axis]
            errs = (obj.avg_force_v_pos[key][resp_axis,0][2]) * cal_facs[resp_axis]
            axarr[resp_axis,0].errorbar(xdat, ydat*1e15, errs*1e15, \
                              label = lab, fmt='.-', ms=10, color = col)

            xdat_d = obj.avg_diag_force_v_pos[key][resp_axis,0][0]
            ydat_d = obj.avg_diag_force_v_pos[key][resp_axis,0][1] + offset_d
            errs_d = obj.avg_diag_force_v_pos[key][resp_axis,0][2]
            axarr[resp_axis,1].errorbar(xdat_d, ydat_d*1e15, errs_d*1e15, \
                              label = lab, fmt='.-', ms=10, color = col)

            if subtract_background:
                if i == 0:
                    sub_dat[resp_axis] = np.copy(ydat)
                    sub_dat_d[resp_axis] = np.copy(ydat_d)

                axarr2[resp_axis,0].errorbar(xdat, (ydat-sub_dat[resp_axis])*1e15, errs*1e15, \
                                             label = lab, fmt='.-', ms=10, color = col)
                axarr2[resp_axis,1].errorbar(xdat_d, (ydat_d-sub_dat_d[resp_axis])*1e15, \
                                             errs_d*1e15, \
                                             label = lab, fmt='.-', ms=10, color = col)

if subtract_background:
    arrs = [axarr, axarr2]
else:
    arrs = [axarr,]

for arr in arrs:
    arr[0,0].set_title('Raw Data: X, Y and Z-response')
    arr[0,1].set_title('Diagonalized Data: X, Y and Z-response')

    for col in [0,1]:
        arr[2,col].set_xlabel('Distance from Cantilever [um]')

    arr[0,0].set_ylabel('X-direction Force [fN]')
    arr[1,0].set_ylabel('Y-direction Force [fN]')
    arr[2,0].set_ylabel('Z-direction Force [fN]')

    arr[0,0].legend(loc=0, numpoints=1, ncol=2, fontsize=9)

if len(fig_title):
    f.suptitle(fig_title, fontsize=18)

plt.show()
