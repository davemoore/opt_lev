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

dirs = [155,]

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )
#print ddict

load_from_file = False
cant_axis = 2
step_axis = 0
respaxis = 1
bin_size = 5

init_data = [0., 0., 20.]
load_charge_cal = True

fit_height = True
fit_dist = 30.   # um

maxfiles = 1000

fig_title = 'Force vs. Cantilever Position: Finding height'

tf_path = './trans_funcs/Hout_20160715.p'
step_cal_path = './calibrations/step_cal_20160715.p'

#################

def ffn(x, a, b):
    return a * (1. / x)**2 + b * (1. / x)

def ffn2(x, a, b, c):
    return a * (x - b)**2 + c



def proc_dir(d):
    dv = ddict[d]

    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)


pos_dict = {}
for obj in dir_objs:
    for fobj in obj.fobjs:
        cpos = fobj.get_stage_settings(axis=step_axis)[0]
        cpos = cpos * 80. / 10.   # 80um travel per 10V control
        if cpos not in pos_dict:
            pos_dict[cpos] = []
            pos_dict[cpos].append(fobj.fname)
        else:
            pos_dict[cpos].append(fobj.fname)


colors = bu.get_color_map(len(pos_dict.keys()))

pos_keys = pos_dict.keys()
pos_keys.sort()

force_at_closest = {}
fits = {}
diag_fits = {}

f, axarr = plt.subplots(3,2,sharex='all',sharey='all',figsize=(10,12),dpi=100)
for i, pos in enumerate(pos_keys):
    newobj = cu.Data_dir(0, init_data, pos)
    newobj.files = pos_dict[pos]
    newobj.load_dir(cu.diag_loader, maxfiles=maxfiles)
    newobj.get_avg_force_v_pos(cant_axis=cant_axis, bin_size = bin_size)

    #newobj.load_H("./trans_funcs/Hout_20160630.p")
    newobj.load_H(tf_path)
    #newobj.plot_H(show=True)

    if load_charge_cal:
        #newobj.load_step_cal('./calibrations/step_cal_20160628.p') 
        newobj.load_step_cal(step_cal_path)
    else:
        newobj.charge_step_calibration = step_calibration

    #newobj.get_conv_facs()

    newobj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=200., # plot_Happ=True, \
                             build_conv_facs=True, drive_freq=18.)
    newobj.get_avg_diag_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)


    keys = newobj.avg_diag_force_v_pos.keys()
    cal_facs = newobj.conv_facs
    #cal_facs = [1.,1.,1.]
    color = colors[i]
    posshort = '%0.2f' % float(pos)
    for key in keys:
        diagdat = newobj.avg_diag_force_v_pos[key]
        dat = newobj.avg_force_v_pos[key]
        #offset = 0
        lab = posshort + ' um'
        for resp in [0,1,2]:
            offset = - dat[resp,0][1][-1]
            diagoffset = - diagdat[resp,0][1][-1]
            axarr[resp,0].errorbar(dat[resp,0][0], \
                                   (dat[resp,0][1]+offset)*cal_facs[resp]*1e15, \
                                   dat[resp,0][2]*cal_facs[resp]*1e15, \
                                   fmt='.-', ms=10, color = color, label=lab)
            axarr[resp,1].errorbar(diagdat[resp,0][0], \
                                   (diagdat[resp,0][1]+diagoffset)*1e15, \
                                   diagdat[resp,0][2]*1e15, \
                                   fmt='.-', ms=10, color = color, label=lab)

        if fit_height:
            offset = -dat[respaxis,0][1][-1]
            diagoffset = -diagdat[respaxis,0][1][-1]
            popt, pcov = curve_fit(ffn, dat[respaxis,0][0], \
                                           (dat[respaxis,0][1]+offset)*cal_facs[respaxis]*1e15, \
                                           p0=[1.,0.1])
            diagpopt, diagpcov = curve_fit(ffn, diagdat[respaxis,0][0], \
                                           (diagdat[respaxis,0][1]+diagoffset)*1e15, \
                                           p0=[1.,0.1])

            fits[pos] = (popt, pcov)
            diag_fits[pos] = (diagpopt, diagpcov)

axarr[0,0].set_title('Raw Data: X, Y and Z-response')
axarr[0,1].set_title('Diagonalized Data: X, Y and Z-response')

for col in [0,1]:
    axarr[2,col].set_xlabel('Distance from Cantilever [um]')

axarr[0,0].set_ylabel('X-direction Force [fN]')
axarr[1,0].set_ylabel('Y-direction Force [fN]')
axarr[2,0].set_ylabel('Z-direction Force [fN]')

axarr[0,0].legend(loc=0, numpoints=1, ncol=2, fontsize=9)

if len(fig_title):
    f.suptitle(fig_title, fontsize=18)



if fit_height:
    keys = fits.keys()
    keys.sort()
    keys = map(float, keys)
    arr1 = []
    arr2 = []
    for key in keys:
        arr1.append(ffn(fit_dist, fits[key][0][0], fits[key][0][1]))
        arr2.append(ffn(fit_dist, diag_fits[key][0][0], diag_fits[key][0][1]))

    diff1 = np.abs(np.amax(arr1) - np.amin(arr1))
    diff2 = np.abs(np.amax(arr2) - np.amin(arr2))

    p0_1 = [diff1, 40, diff1]
    p0_2 = [diff2, 40, diff2]

    fit1, err1 = curve_fit(ffn2, keys, arr1, p0 = p0_1)
    fit2, err2 = curve_fit(ffn2, keys, arr2, p0 = p0_2)
    xx = np.linspace(keys[0], keys[-1], 100)
    fxx1 = ffn2(xx, fit1[0], fit1[1], fit1[2]) 
    fxx2 = ffn2(xx, fit2[0], fit2[1], fit2[2]) 

    plt.figure()
    plt.suptitle("Fit of Raw Data")
    plt.plot(keys, arr1)
    plt.plot(xx, fxx1)
    plt.figure()
    plt.suptitle("Fit of Diagonalized Data")
    plt.plot(keys, arr2)
    plt.plot(xx, fxx2)

    print "Best fit positions: ", fit1[1], fit2[1]
    
plt.show()
    
        
