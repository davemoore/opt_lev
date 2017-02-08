import numpy as np
import matplotlib.pyplot as plt


drive_elecs = [1, 3]

voltages = [0.1, 1]
freqs = [18, 18]
Fsamp = 5000.
cutsamp = 2000
Npoints = 250000

out_fil = r'C:\GitHub\opt_lev\labview\DAQ_settings\known_force_6.txt'

######################################


dt = 1. / Fsamp
t = np.linspace(0, (Npoints-1) * dt, Npoints)

out_arr = []

for ind in range(8):
    out_arr.append(np.zeros(Npoints))
    for ind2 in range(len(drive_elecs)):
        if ind == drive_elecs[ind2]:
            drive = voltages[ind2] * np.sin(2 * np.pi * freqs[ind2] * t)
            out_arr[ind] += drive

out_arr = np.array(out_arr)
#print out_arr.shape

cheat = np.c_[drive_elecs, freqs, voltages]

np.savetxt(out_fil, out_arr, fmt='%.8e', delimiter=",")
