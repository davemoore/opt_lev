import matplotlib.pyplot as plt
import os
import numpy as np



path = "/home/arider/analysis/20140711/Bead6/cal_charge_chirps_100s"

freqs = np.load(os.path.join(path, 'Hfreqs.npy'))
H = np.load(os.path.join(path, 'H.npy'))


fig = plt.figure()
plt.subplot(2, 1, 1)
plt.loglog(freqs, np.abs(H)**2, 'r.')
plt.ylabel('|H|')
plt.subplot(2, 1, 2)
plt.semilogx(freqs, np.angle(H), 'b.')
plt.ylabel('Phase[rad]')
plt.xlabel("Frequency[Hz]")
plt.show()
