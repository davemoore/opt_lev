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
from multiprocessing import Pool

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

fig_title = 'Force vs. Cantilever Position: 18 Hz, Helium, HERMES-160808'
setylim = False
ylim = [-2.5,13.5]

plot_log_scale = True
logylim = [0.005, 100]

exp_approx = True
fit_portion = 1.0
cant_throw = 80.

tf_path = './trans_funcs/Hout_20160808.p'
step_cal_path = './calibrations/step_cal_20160808.p'

def fit_fun(x, a, b, c):
    return (a * np.exp( b * x))*x + c

####################################################
##### Data Directories, Reverse Chronological ######

#background_dirs = [411, 412, 413, 414, 415, 416]        # Kr Background
background_dirs = [423, 424, 425, 426, 427, 428]         # He Background
#background_dirs = [437, 438, 439, 440, 441]             # Ar Background
#background_dirs = [449, 450, 451, 452, 453, 454]        # Xe Background
#background_dirs = [461, 462, 463, 464, 465]             # Xe-2 Background
#background_dirs = [472, 473, 474, 475]                  # Kr-2 Background
#background_dirs = [482, 483, 484, 485, 486, 487, 488,]  # Ar-2 Background
#background_dirs = [496, 497, 498, 499, 500]             # He-2 Background
use_endpoint = True

#### New Gas Handling System

# Bead 1, 8-05 calibrations
#dirs = [393, 394, 395, 396, 397, 398] # 17 Hz with Kr


# Bead 2, 8-08 calibrations

#dirs = [413, 417, 418, 419, 420, 421, 422]          # 18 Hz with Kr
dirs = [428, 430, 431, 432, 433, 434, 435,]# 436]   # 18 Hz with He
#dirs = [438, 442, 443, 444, 445, 446, 447,]#448]    # 18 Hz with Ar
#dirs = [453, 455, 456, 457, 458, 459, 460]          # 18 Hz with Xe
#dirs = [462, 466, 467, 468, 469, 470, 471]          # 18 Hz with Xe - Repeat
#dirs = [472, 476, 477, 478, 479, 480, 481]          # 18 Hz with Kr - Repeat
#dirs = [482, 489, 490, 491, 492, 493, 494,]#495]    # 18 Hz with Ar - Repeat
#dirs = [497, 501, 502, 503, 504, 505, 506,]#507]    # 18 Hz with He - Repeat



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
    else:
        dir_obj.charge_step_calibration = step_calibration

    dir_obj.calibrate_H()

    dir_obj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=200.,# plot_Happ=True, \
                             build_conv_facs=True, drive_freq=18.)

    return dir_obj


def diagonalize(dir_obj):
    dir_obj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=200.,# plot_Happ=True, \
                             build_conv_facs=True, drive_freq=18.)

dir_objs = map(proc_dir, dirs)

background_dir_objs = map(proc_dir, background_dirs)

#pool = Pool(5)
#pool.map(diagonalize, dir_objs)
#pool.map(diagonalize, background_dir_objs)

thermal_cal_file_path = '/data/20160808/bead1/1_5mbar_zcool_final.h5'

xdat_background = []
backgrounds = [[], [], []]
background_errs = [[], [], []]

backgrounds_d = [[], [], []]
background_errs_d = [[], [], []]

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

            backgrounds[resp_axis].append(ydat)
            backgrounds_d[resp_axis].append(ydat_d)
            counts[resp_axis] += 1

background_signs_d = [0.,0.,0.]
for resp_axis in [0,1,2]:
    background_errs[resp_axis] = np.std(backgrounds[resp_axis], axis=0)
    backgrounds[resp_axis] = np.mean(backgrounds[resp_axis], axis=0)
    background_errs_d[resp_axis] = np.std(backgrounds_d[resp_axis], axis=0)
    backgrounds_d[resp_axis] = np.mean(backgrounds_d[resp_axis], axis=0)
    
    if backgrounds_d[resp_axis][0] < backgrounds_d[resp_axis][-1]:
        background_signs_d[resp_axis] = -1.0
    else:
        background_signs_d[resp_axis] = 1.0

colors_yeay = bu.get_color_map( len(dir_objs) )
f, axarr = plt.subplots(3,2,sharey='all',sharex='all',figsize=(10,12),dpi=100)

if subtract_background:
    f2, axarr2 = plt.subplots(3,2,sharey='all',sharex='all',figsize=(10,12),dpi=100)
    sub_dat = [[],[],[]]
    sub_dat_d = [[],[],[]]
    fig_title_d = fig_title + ', Subtracted'

    fb, axb = plt.subplots(3,1, sharex='all', sharey='all')
    ffit, axfit = plt.subplots(1,2, figsize=(8,10), dpi=100)
    for resp_axis in [0,1,2]:
        off = backgrounds_d[resp_axis][-1]
        axb[resp_axis].errorbar(xdat_background, (backgrounds_d[resp_axis]+off)*1e15, \
                                  background_errs_d[resp_axis]*1e15, fmt='.-')

if plot_log_scale:
    f3, ax3 = plt.subplots(figsize=(10,8), dpi=100)
    ax3.set_yscale('log')
    #off = backgrounds_d[resp_axis][-1]
    off = 0.
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

            offset = 0.
            offset_d = 0.

            if exp_approx and resp_axis == 1:
                # Fit portion to exponential -> Extrapolate for offset
                endpoint = xdat[-1] - cant_throw * (1 - fit_portion)
                inds = xdat < endpoint
                
                p0 = [1e-14, -0.01, 0]

                popt, pcov = curve_fit(fit_fun, xdat[inds], ydat[inds], p0=p0, maxfev=10000)
                popt_d, pcov_d = curve_fit(fit_fun, xdat_d[inds], ydat_d[inds], p0=p0, maxfev=10000) 

                #print popt
                #print popt_d

                offset = -1.0 * popt[-1]
                offset_d = -1.0 * popt_d[-1]

                thresh_up = 5. * (np.max(ydat*1e15) - np.min(ydat*1e15))
                thresh_up_d = 5. * (np.max(ydat_d*1e15) - np.min(ydat_d*1e15))
                thresh_down = (2. / 5.) * thresh_up
                thresh_down_d = (2. / 5.) * thresh_up_d

                if np.abs(offset*1e15) > thresh_up:
                    offset = 0
                elif np.abs(offset*1e15) > thresh_down:
                    offset = 0.25 * thresh_down * 1e-15

                if np.abs(offset_d*1e15) > thresh_up_d:
                    offset_d = 0
                elif np.abs(offset_d*1e15) > thresh_down_d:
                    offset_d = 0.25 * thresh_down_d * 1e-15

                fitpoints = fit_fun(xdat, *popt)
                axfit[0].plot(xdat, fitpoints*1e15, color = col)
                axfit[0].errorbar(xdat, ydat*1e15, errs*1e15, \
                                  fmt='.-', color = col)

                fitpoints_d = fit_fun(xdat_d, *popt_d)
                axfit[1].plot(xdat_d, fitpoints_d*1e15, color = col)
                axfit[1].errorbar(xdat_d, ydat_d*1e15, errs_d*1e15, \
                              fmt='.-', color = col)

            if subtract_background:
                errs = np.sqrt( errs**2 + background_errs[resp_axis]**2 )
                errs_d = np.sqrt( errs_d**2 + background_errs_d[resp_axis]**2 )

                axarr2[resp_axis,0].errorbar(xdat, (ydat-offset)*1e15, errs*1e15, \
                                             label = lab, fmt='.-', ms=10, color = col)
                axarr2[resp_axis,1].errorbar(xdat_d, (ydat_d-offset_d)*1e15, errs_d*1e15, \
                                             label = lab, fmt='.-', ms=10, color = col)

            if plot_log_scale and resp_axis == 1:
                ax3.errorbar(xdat_d, np.abs(ydat_d-offset_d)*1e15, \
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