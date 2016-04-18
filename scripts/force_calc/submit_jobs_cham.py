import os, time
import numpy as np

lam_list = np.logspace(0,2.0,20)*1e-6
#print lam_list

for j in range(len(lam_list)):
    cc = "python force_calc_cham_geom.py %e" % lam_list[j]
    print cc
    os.system(cc)

