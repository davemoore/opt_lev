import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seismic_noise as sn

path161 = '/data_slave/20140908/room161_data'
pathes = '/data/20140910/end_station_data'
pathbead = '/data/20140912/Bead1/turbo_down'

dat161, freqs = sn.getpsd_ave(path161)
dates, freqs = sn.getpsd_ave(pathes)
datbd, freqs = sn.getpsd_ave(pathbead, c1 = -1, fmax = 200)

dat161 /= (50*629)**2
dates /= (50*629)**2
datbd /= (100*629)**2

np.save('dat161.npy', dat161)
np.save('dates.npy', dates)
np.save('datbd.npy', datbd)
np.save('freqs.npy', freqs)
