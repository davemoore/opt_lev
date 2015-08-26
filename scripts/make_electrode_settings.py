import numpy as np

#################################################
fname = r"..\labview\DAQ_settings\electrode_sweep.txt"


## dc offsets to sweep over
dc_list = np.linspace(-4., 4., 10 ) ## V

## list of electrode frequencies
freq_list = np.array([13, 17, 19, 23, 29, 31, 37, 41]) ## Hz

## list of drive amplitudes
drive_amp = 0.5 ## V
amp_list = drive_amp*np.ones_like(freq_list)
##################################################

par_list = []
for dc in dc_list:
    electrodes_to_use = 1.0*(freq_list > 0.)
    dc_list = dc * electrodes_to_use
    par_list.append( np.hstack( [electrodes_to_use, amp_list, freq_list, dc_list] ) )

par_list = np.array(par_list)

np.savetxt(fname, par_list, delimiter=",", fmt="%.2f")
