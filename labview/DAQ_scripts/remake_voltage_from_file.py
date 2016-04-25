## load a file containing values of the voltage vs beta and remake it, assuming
## that the force scales like the square of the voltage

import numpy as np
import matplotlib.pyplot as plt

inbeta = 3e8
infile = r"C:\Data\20160320\bead1\cham_force_beta_3e+08.txt"
outdir = r"C:\Data\20160320\bead1\cham_force_files"
olddat = np.loadtxt(infile,delimiter=',')

beta_list = [5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8]

plt.figure()
for b in beta_list:

    curr_dat = olddat*1.0
    curr_dat[:,1] = np.sqrt(b/inbeta)*curr_dat[:,1]

    output_file = outdir + "\cham_force_beta_%.0e.txt"%b
    print "Writing " + output_file
    np.savetxt( output_file, curr_dat, delimiter="," )

    plt.semilogy( curr_dat[:,0], curr_dat[:,1], label="%.1e"%b )

plt.legend()
plt.show()
