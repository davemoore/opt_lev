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

####################################################
####### Input parameters for data processing #######

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )
#print ddict

respdir = 'Y'
resp_axis = 1
cant_axis = 2
bin_size = 5

load_charge_cal = True
maxfiles = 1000

subtract_background = True

fig_title = 'Force vs. Cantilever Position: 18 Hz, Argon - 2, HERMES-160808'
setylim = False
ylim = [-2.5,13.5]

plot_log_scale = True
logylim = [0.005, 100]
exp_approx = True
cant_throw = 80.

tf_path = './trans_funcs/Hout_20160808.p'
step_cal_path = './calibrations/step_cal_20160808.p'


####################################################
##### Data Directories, Reverse Chronological ######

background_dirs = [482, 483, 484, 485, 486, 487, 488,] # Ar-2 Background
#background_dirs = [498, 499,] # He-2 Background
use_endpoint = True

#### New Gas Handling System

# Bead 1, 8-05 calibrations
#dirs = [393, 394, 395, 396, 397, 398] # 17 Hz with Kr


# Bead 2, 8-08 calibrations

#dirs = [413, 417, 418, 419, 420, 421, 422] # 18 Hz with Kr
#dirs = [428, 430, 431, 432, 433, 434, 435, 436] # 18 Hz with He
#dirs = [438, 442, 443, 444, 445, 446, 447,]#448] # 18 Hz with Ar
#dirs = [452, 455, 456, 457, 458, 459, 460] # 18 Hz with Xe
#dirs = [462, 466, 467, 468, 469, 470, 471] # 18 Hz with Xe - Repeat
#dirs = [472, 476, 477, 478, 479, 480, 481] # 18 Hz with Kr - Repeat
dirs = [482, 489, 490, 491, 492, 493, 494,]#495] # 18 Hz with Ar - Repeat
#dirs = [497, 501, 502, 503, 504, 505, 506,]#507] # 18 Hz with He - Repeat



#### Series of Noble Gas Measurements
# use 7-27 calibrations

#dirs = [311,316,317,318,319,320,321,322]   # 18 Hz with He
#dirs = [323,328,329,330,331,332,333,334]   # 1.5 Hz with He

#dirs = [340,347,348,349,350,]#351,352,353]   # 18 Hz with Ar
#dirs = [354,361,362,363,364,365,366,367]   # 1.5 Hz with Ar



#### First Good Data Sets

#dirs = [220,222,223,224,225,226,227,]#228]   # 18 Hz with N2
#dirs = [229,232,233,234,235,236,237,]#238]   # 1.5 Hz with N2

#dirs = [240,242,243,244,245,246,247,]#248]   # 18 Hz with He
#dirs = [250,252,253,254,255,256,257,]#258]   # 1.5 Hz with He

#dirs = [261,262,263,264,265,266,267,]#228]   # 18 Hz with N2


############




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

    dir_obj.calibrate_H()

    dir_obj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=200.,# plot_Happ=True, \
                             build_conv_facs=True, drive_freq=18.)

    return dir_obj

dir_objs = map(proc_dir, dirs)

background_dir_objs = map(proc_dir, background_dirs)

thermal_cal_file_path = '/data/20160805/bead1/1_5mbar_zcool_final.h5'

xdat_background = []
backgrounds = [[], [], []]
background_errs = [[], [], []]

backgrounds_d = [[], [], []]
background_errs_d = [[], [], []]
background_offsets_d = [0.,0.,0.]

counts = [0., 0., 0.]
for i, obj in enumerate(background_dir_objs):

    obj.get_avg_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)

    obj.get_avg_diag_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)

    keys = obj.avg_force_v_pos.keys()
    cal_facs = obj.conv_facs
    for key in keys:
        for resp_axis in [0,1,2]:
            xdat = obj.avg_force_v_pos[key][resp_axis,0][0]
            ydat = (obj.avg_force_v_pos[key][resp_axis,0][1]) * cal_facs[resp_axis]
            errs = (obj.avg_force_v_pos[key][resp_axis,0][2]) * cal_facs[resp_axis]

            xdat_d = obj.avg_diag_force_v_pos[key][resp_axis,0][0]
            ydat_d = obj.avg_diag_force_v_pos[key][resp_axis,0][1]
            errs_d = obj.avg_diag_force_v_pos[key][resp_axis,0][2]

            if not len(xdat_background):
                xdat_background = xdat

            if not len(backgrounds[resp_axis]):
                backgrounds[resp_axis] = ydat
                background_errs[resp_axis] = errs
                backgrounds_d[resp_axis] = ydat_d
                background_errs_d[resp_axis] = errs_d
                counts[resp_axis] += 1
            else:
                backgrounds[resp_axis] += ydat
                background_errs[resp_axis] += errs
                backgrounds_d[resp_axis] += ydat_d
                background_errs_d[resp_axis] += errs_d
                counts[resp_axis] += 1.

background_signs_d = [0.,0.,0.]
for resp_axis in [0,1,2]:
    backgrounds[resp_axis] = np.array(backgrounds[resp_axis]) / counts[resp_axis]
    background_errs[resp_axis] = np.array(background_errs[resp_axis]) / counts[resp_axis]
    backgrounds_d[resp_axis] = np.array(backgrounds_d[resp_axis]) / counts[resp_axis]
    background_errs_d[resp_axis] = np.array(background_errs_d[resp_axis]) / counts[resp_axis]
    
    if backgrounds_d[resp_axis][0] < backgrounds_d[resp_axis][-1]:
        background_signs_d[resp_axis] = -1.0
    else:
        background_signs_d[resp_axis] = 1.0

for resp_axis in [0,1,2]:
    if use_endpoint:
        background_offsets_d[resp_axis] = -1.0 * backgrounds_d[resp_axis][-1]
    else:
        if background_signs_d[resp_axis] < 0:
            background_offsets_d[resp_axis] = -1.0 * np.amax(backgrounds_d[resp_axis])
        else:
            background_offsets_d[resp_axis] = -1.0 * np.amin(backgrounds_d[resp_axis])
            

colors_yeay = bu.get_color_map( len(dir_objs) )
f, axarr = plt.subplots(3,2,sharey='all',sharex='all',figsize=(10,12),dpi=100)

if subtract_background:
    f2, axarr2 = plt.subplots(3,2,sharey='all',sharex='all',figsize=(10,12),dpi=100)
    sub_dat = [[],[],[]]
    sub_dat_d = [[],[],[]]
    fig_title_d = fig_title + ', Subtracted'

    fb, axb = plt.subplots(3,1, sharex='all', sharey='all')
    for resp_axis in [0,1,2]:
        off = background_offsets_d[resp_axis]
        axb[resp_axis].errorbar(xdat_background, (backgrounds_d[resp_axis]+off)*1e15, \
                                  background_errs_d[resp_axis]*1e15, fmt='.-')

if plot_log_scale:
    f3, ax3 = plt.subplots(figsize=(10,8), dpi=100)
    ax3.set_yscale('log')
    off = background_offsets_d[1]
    ax3.errorbar(xdat_background, np.abs(backgrounds_d[1]+off)*1e15, \
                 background_errs_d[1]*1e15, \
                 fmt = '--', label = 'Background', color = 'k')

for i, obj in enumerate(dir_objs):
    col = colors_yeay[i]
    cal_facs = obj.conv_facs

    obj.get_avg_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)

    obj.get_avg_diag_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)

    keys = obj.avg_force_v_pos.keys()
    for key in keys:
        lab = '%g mbar' %(cu.round_sig(obj.avg_pressure[2],sig=2))

        for resp_axis in [0,1,2]:

            xdat = obj.avg_force_v_pos[key][resp_axis,0][0]
            ydat = (obj.avg_force_v_pos[key][resp_axis,0][1]) * cal_facs[resp_axis]
            errs = (obj.avg_force_v_pos[key][resp_axis,0][2]) * cal_facs[resp_axis]
            axarr[resp_axis,0].errorbar(xdat, (ydat-ydat[-1])*1e15, errs*1e15, \
                              label = lab, fmt='.-', ms=10, color = col)

            xdat_d = obj.avg_diag_force_v_pos[key][resp_axis,0][0]
            ydat_d = obj.avg_diag_force_v_pos[key][resp_axis,0][1]
            errs_d = obj.avg_diag_force_v_pos[key][resp_axis,0][2]
            axarr[resp_axis,1].errorbar(xdat_d, (ydat_d-ydat_d[-1])*1e15, errs_d*1e15, \
                              label = lab, fmt='.-', ms=10, color = col)

            if subtract_background:
                ydat = ydat - backgrounds[resp_axis]
                ydat_d = ydat_d - backgrounds_d[resp_axis]

            if resp_axis == 1:
                offset = -1.0 * ydat[-1]
                offset_d = -1.0 * ydat_d[-1]
            elif resp_axis != 1:
                offset = 0.
                offset_d = 0.

            if exp_approx:
                # Fit first half to exponential -> Extrapolate for offset
                # F = A exp[-k x]  =>  Ffar = Fclose * exp[-k (xfar - xclose)]
                ks  = []
                ks_d = []
                midpoint = np.mean(xdat)

                for i, point1 in enumerate(xdat):
                    for j, point2 in enumerate(xdat):
                        if point1 > midpoint or point2 > midpoint:
                            continue
                        if point1 > point2:
                            continue

                    k = -1.0 * np.log(ydat[j] / ydat[i]) / (point2 - point1)
                    k_d = -1.0 * np.log(ydat_d[j] / ydat_d[i]) / (point2 - point1)
                    ks.append(k)
                    ks_d.append(k_d)

                kavg = np.mean(ks)
                kavg_d = np.mean(k_d)
                
                if resp_axis == 1:
                    offset += ydat[0] * np.exp(-1.0 * kavg * cant_throw)
                    offset_d += ydat_d[0] * np.exp(-1.0 * kavg * cant_throw)

            if subtract_background:

                axarr2[resp_axis,0].errorbar(xdat, ydat*1e15, errs*1e15, \
                                             label = lab, fmt='.-', ms=10, color = col)
                axarr2[resp_axis,1].errorbar(xdat_d, ydat_d*1e15, \
                                             errs_d*1e15, label = lab, fmt='.-', ms=10, color = col)

            if plot_log_scale and resp_axis == 1:
                off = background_offsets_d[resp_axis]
                ax3.errorbar(xdat_d, np.abs(ydat_d-backgrounds_d[resp_axis]+off)*1e15, \
                             errs_d*1e15, fmt = '.-', label = lab, color = col)

if plot_log_scale:
    ax3.set_xlabel('Distance from Cantilever [um]')
    ax3.set_ylabel('Y-direction Force [fN]')
    ax3.legend(loc=0, numpoints=1, ncol=2, fontsize=9)
    f3.suptitle(fig_title_d, fontsize=18)
    ax3.set_ylim(logylim)

if subtract_background:
    arrs = [axarr, axarr2]
else:
    arrs = [axarr,]

for arr in arrs:
    arr[0,0].set_title('Raw Imaging Response')
    arr[0,1].set_title('Diagonalized Forces')

    for col in [0,1]:
        arr[2,col].set_xlabel('Distance from Cantilever [um]')

    arr[0,0].set_ylabel('X-direction Force [fN]')
    arr[1,0].set_ylabel('Y-direction Force [fN]')
    arr[2,0].set_ylabel('Z-direction Force [fN]')

    arr[0,0].legend(loc=0, numpoints=1, ncol=2, fontsize=9)

if setylim:
    axarr[0,0].set_ylim(ylim)
    if subtract_background:
        axarr2[0,0].set_ylim(ylim)

if len(fig_title):
    f.suptitle(fig_title, fontsize=18)
    if subtract_background:
        f2.suptitle(fig_title_d, fontsize=18)

plt.show()
