import cant_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import Tkinter
import tkFileDialog
import os




data_dir = "/data/20160320/bead1"

calf = '/data/20160320/bead1/1_5mbar_zcool.h5'


dict = bu.load_dir_file('dir_file.txt')
files = glob.glob(dict['215'][0] + "/*.h5")

f1 = cu.data()
f1.load(calf, [0, 0, 20])

f1.psd()

popt, pcov = cu.thermal_fit(f1.psds[0], f1.psd_freqs)

plt.loglog(f1.psd_freqs, f1.psds[0])
plt.loglog(f1.psd_freqs, cu.thermal_psd_spec(f1.psd_freqs, popt[0], popt[1], popt[2]) , 'r')
plt.show()


#A, f0, g. A = vpmsq*2.*kb*T/mb

vpmsq = popt[0]*bu.bead_mass/(2.*bu.kb*300.)
k = bu.bead_mass*(2.*np.pi*popt[1])**2
npv = k/np.sqrt(vpmsq)
print npv
