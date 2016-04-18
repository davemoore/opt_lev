import matplotlib.pyplot as plt
import numpy as np
import bead_util as bu


xx = np.linspace(2, 200, 1e3)

f=plt.figure()
plt.semilogy( xx, 1e17*np.abs(bu.get_es_force(xx*1e-6, volt=1.0)), 'k', linewidth=1.5, label="Induced" )
plt.semilogy( xx, 1e17*np.abs(bu.get_es_force(xx*1e-6, volt=1.0, is_fixed=True)), 'r', linewidth=1.5, label="Permanent" )

plt.xlabel("Distance [$\mu$m]")
plt.ylabel("Force, arbitrary scaling")
plt.xlim([0,100])
plt.ylim([1, 1e3])
plt.legend()
f.set_size_inches(6,4.5)
plt.subplots_adjust(bottom=0.12, top=0.95, right=0.95)
plt.savefig("ebkg_scaling.pdf")
plt.show()
