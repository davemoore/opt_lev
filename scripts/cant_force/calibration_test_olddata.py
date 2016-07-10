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


#cal = [['/data/20160329/bead1/chargelp_cal2'], 'Cal', 20, 1e-13]
cal = [['/data/20160403/bead1/chargelp_cal2'], 'Cal', 20, 1e-13]

cal_dir_obj = cu.Data_dir(cal[0], [0,0,cal[2]], cal[1])
cal_dir_obj.load_dir(cu.simple_loader)
cal_dir_obj.build_step_cal_vec()

#print cal_dir_obj.Hs

cal_dir_obj.step_cal(cal_dir_obj)

cal_dir_obj.load_H("./trans_funcs/Hout_20160613.p")
cal_dir_obj.calibrate_H()

cal_dir_obj.get_conv_facs()


test_data_obj = cu.Data_file()
test_data_obj.load("/data/20160403/bead1/cant_sweep_20um/urmbar_xyzcool_stageX3500nmY5000nmZ5000nmZ5000mVAC18Hz_5.h5", [0,0,20])


N = np.shape(test_data_obj.pos_data)[1]

test_data_obj.get_fft()

test_data_obj.ms()

data_psd = np.abs(test_data_obj.data_fft)**2 * 2./(N * test_data_obj.Fsamp)

test_data_obj.spatial_bin()

plt.figure()
for i in [0,1,2]:
    plt.subplot(3,1,i+1)
    plt.plot(test_data_obj.binned_cant_data[0][i][2], \
             test_data_obj.binned_pos_data[0][i][2]*cal_dir_obj.conv_facs[i])

plt.figure()
for i in [0,1,2]:
    plt.subplot(3,1,i+1)
    plt.loglog(test_data_obj.fft_freqs, np.sqrt(data_psd[i]) * cal_dir_obj.conv_facs[i])




##### Compare to thermal calibration

cal_obj = cu.Data_file()
cal_obj.load("/data/20160403/bead1/1_5mbar_zcool.h5", [0,0,20])

cal_obj.thermal_calibration()
cal_obj.plt_thermal_fit()

norm_rats = cal_obj.get_thermal_cal_facs()

print "Charge Calibration"
print cal_dir_obj.conv_facs
print
print "Thermal Calibration"
print [np.sqrt(norm_rats[0]), np.sqrt(norm_rats[1]), np.sqrt(norm_rats[2])]



