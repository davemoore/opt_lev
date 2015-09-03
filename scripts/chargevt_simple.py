import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import os
import glob

path = "/data/20150119/Bead6/chargelp"

reprocess_file = True

def get_data(fname):
    dat, attribs, f = bu.getdata(fname)
    corr = bu.good_corr(dat[:, 0], dat[:, -1], attribs['Fsamp'], 41.)
    f.close()
    return corr[0], np.max(corr)

if reprocess_file:
    init_list = glob.glob(path + "/*500mV*.h5")
    files = sorted(init_list, key = bu.find_str)
    no_offset = []
    maxc = []
    for f in files[::]:
        try:
            print f
            corri, maxi = get_data(f)
            no_offset.append(corri)
            maxc.append(maxi)
        except:
            print "probably bad file"

plt.plot(no_offset)
plt.plot(maxc)
plt.xlabel('file #')
plt.ylabel("correlation [V^2]")
plt.show()
    
