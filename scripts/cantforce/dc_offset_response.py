import numpy as np
import cPickle as pickle
import bead_util as bu
import matplotlib.pyplot as plt

datadict = pickle.load( open('testout.p', 'rb'))

keylist = datadict.keys()

del keylist[keylist.index('bins')]

keylist.sort()

bins = datadict['bins']

colors_yeay = bu.get_color_map(len(keylist) / 3)

k = 0
for key in keylist:
    if 'z' not in key:
        continue
    plt.plot(bins, datadict[key] - datadict['0.0 Vz'], color = colors_yeay[k])
    k += 1

plt.show()
    
