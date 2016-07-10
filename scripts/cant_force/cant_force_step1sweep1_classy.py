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

dirs = [65,]

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )
#print ddict

load_from_file = False
cant_axis = 1
step_axis = 2
respaxis = 1
init_data = [0., 0., 0]
load_charge_cal = True

fit_height = False

maxfiles=1000

#################

def ffn(x, a, b):
    return a * (1. / x) + b

def ffn2(x, a, b, c):
    return a * (x - b)**2 + c

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

for i, pos in enumerate(pos_keys):
    newobj = cu.Data_dir(0, init_data, pos)
    newobj.files = pos_dict[pos]
    newobj.load_dir(cu.diag_loader, maxfiles=maxfiles)
    newobj.get_avg_force_v_pos(axis = respaxis, \
                               cant_axis=cant_axis, bin_size = 4)

    newobj.load_H("./trans_funcs/Hout_20160630.p")
    #newobj.plot_H(show=True)

    if load_charge_cal:
        newobj.load_step_cal('./calibrations/step_cal_20160701.p')
    else:
        newobj.charge_step_calibration = step_calibration

    newobj.calibrate_H()

    newobj.get_conv_facs()

    newobj.diagonalize_files()
    newobj.get_avg_diag_force_v_pos(axis = respaxis, \
                                    cant_axis = cant_axis, bin_size = 4)


    keys = newobj.avg_diag_force_v_pos.keys()
    #cal_facs = newobj.conv_facs
    cal_facs = [1.,1.,1.]
    color = colors[i]
    posshort = '%0.2f' % float(pos)
    for key in keys:
        dat = newobj.avg_diag_force_v_pos[key]
        offset = - dat[1][-1]
        #offset = 0
        lab = posshort + ' um'
        xdat = dat[0]
        ydat = (dat[1]+offset)*cal_facs[respaxis]
        plt.errorbar(dat[0], (dat[1]+offset)*1e15, dat[2]*1e15, fmt='.-', ms=10, color = color, label=lab)
        plt.xlabel('Distance from Cantilever [um]')
        plt.ylabel('Force [fN]')
        ind = np.argmin(dat[0])

        if fit_height:
            popt, pcov = curve_fit(ffn, xdat, ydat, p0=[1.,0])

            fits[pos] = (popt, pcov)
            force_at_closest[pos] = ydat[ind]

plt.legend(loc=0, numpoints=1)

if fit_height:
    keys = fits.keys()
    keys.sort()
    keys = map(float, keys)
    arr1 = []
    arr2 = []
    arr3 = []
    for key in keys:
        arr1.append(force_at_closest[key])
        arr2.append(ffn(30., fits[key][0][0], fits[key][0][1]))
        arr3.append(np.sqrt(fits[key][1][0,0]))
    errs = np.array(arr3) / np.array(keys)
    #arr1 = np.array(arr1)
    #arr2 = np.array(arr2)

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
    plt.plot(keys, arr1)
    plt.plot(xx, fxx1)
    plt.figure()
    plt.errorbar(keys, arr2, errs)
    plt.plot(xx, fxx2)

    print "Best fit positions: ", fit1[1], fit2[1]
    
plt.show()
    
        
