import bead_util as bu
import matplotlib.pyplot as plt
import numpy as np

path = '/data/20141210/Bead3/approach1c'
path1 = '/data/20141210/Bead3/recharged_approach'

avs1c = bu.getforce_ave(path)
avsrc = bu.getforce_ave(path1)[0:-1]
rsx = np.array([3.25, 2.75, 2.25, 1.75, 1.25])*8*5
ry = 70.
rz = 2.*8*5
r = np.sqrt(rsx**2 + ry**2 + rz**2)

plt.plot(r, np.transpose(avs1c)[0],label = '1 charge? x')
plt.plot(r, np.transpose(avsrc)[0],label = 'recharged x')

#plt.plot(r, np.transpose(avs1c)[1], label = '1 charge? y')
#plt.plot(r, np.transpose(avsrc)[1],label = 'recharged y')

#plt.plot(r, np.transpose(avs1c)[2]/50,label = '1 charge? z')
#plt.plot(r, np.transpose(avsrc)[2]/50,label = 'recharged z')

plt.xlabel('distance from tip [um]')
plt.ylabel('force on bead [Riders]')

plt.legend(loc = 4)

plt.show()
