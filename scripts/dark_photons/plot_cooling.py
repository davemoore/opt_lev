import bead_util as bu
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.size'] = 15

fhifb = "/data/20160325/bead1/urmbar_xyzcool_final_spec_hifb.h5"
flowfb = "/data/20160325/bead1/urmbar_xyzcool_lowerxfb4.h5"

fig = plt.figure()

cal = bu.fit_double_peak(flowfb, [192,216], make_plot=True, spars=[1.3e6,205.,0.15],NFFT=2**16,second=False)

cal2 = bu.fit_double_peak(fhifb, [170,200], make_plot=True, spars=[1.e4,184,10],NFFT=2**16,second=True)

plt.legend(loc="upper left", prop={"size": 13})

fig.set_size_inches(6,4.5)
plt.subplots_adjust(bottom=0.125, left=0.14, right=0.95, top=0.95)
plt.savefig("cooling.pdf", dpi=300)

plt.show()
