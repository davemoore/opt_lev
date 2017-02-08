## load a file containing values of the voltage vs beta and remake it, assuming
## that the force scales like the square of the voltage

import numpy as np
import matplotlib.pyplot as plt

#freqs = [5,7,9,13,17,23,29,37,43,51,57,69,71,111,]
#freqs = np.linspace(5,150,10)

np.random.seed()

def choose_freq(low=3, high=80, onFFTbin=False, \
                NFFT = 250000-2000, Fsamp=5000):
    while True:
        freq = np.random.uniform(low, high)
        
        if onFFTbin:
            ft_freqs = np.fft.rfftfreq(NFFT, 1. / Fsamp)
            ind = np.argmin(np.abs(ft_freqs - freq))
            freq = ft_freqs[ind]
            
        if np.abs(freq - 60) > 3:
            break
        
    return freq

numelecs = np.random.choice([2])   # possible driving electrode number
drive_elecs = np.random.choice([1,3,5], numelecs, replace=False)

voltages = []
freqs = []

for i in range(numelecs):
    voltage = np.random.uniform(5,10)
    if drive_elecs[i] == 1:
        voltage = (voltage-5)*0.5
    freq = choose_freq(onFFTbin=False)
    voltages.append(voltage)
    freqs.append(freq)
    

Fsamp = 5000.
cutsamp = 2000
Npoints = 250000


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

np.savetxt(r'C:\GitHub\opt_lev\labview\DAQ_settings\blind_force_4.txt', out_arr, fmt='%.8e', delimiter=",")
np.savetxt(r'C:\GitHub\opt_lev\labview\DAQ_settings\blind_force_4_KEY.txt', cheat, fmt='%.8e', delimiter=",")
    
