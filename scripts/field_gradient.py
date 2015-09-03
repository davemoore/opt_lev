import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


pot = np.loadtxt("/home/dcmoore/comsol/V_vs_x.txt", skiprows = 8)
xcenter = .001015
xspan = 15e-6 
b = np.abs(pot[:, 0] - xcenter) < xspan
x = pot[b, 0]
V = pot[b, 1]
f = UnivariateSpline(x, V, s = 10e-17)
xnew = np.linspace(np.min(x), np.max(x), 10000)

plt.plot(x, V, linewidth = 2, label = "Potential")
plt.plot(xnew, f(xnew), linewidth = 2, label = "Interpolation")
plt.legend()
plt.show()

a = xnew[1]-xnew[0]
E = np.diff(f(xnew))/a

plt.plot(xnew[:-1], E, linewidth = 5, label = 'Electric field')
plt.legend()
plt.xlabel("Distance[um]")
plt.ylabel("E field [V/m]")
plt.show()

dxE = np.diff(E)/a

plt.plot(xnew[:-2], dxE, linewidth = 5, label = 'Electric field gradient')
plt.legend()
plt.xlabel("Distance[um]")
plt.ylabel("E field gradien [V/m^2]")
plt.show()

coupling = E[: -1]*dxE
plt.plot(xnew[:-2], coupling, linewidth = 5, label = 'dipole coupling')
plt.legend()
plt.xlabel("Distance[um]")
plt.ylabel("Dipole coupling [V^2/m^3]")
plt.show()

c10um = coupling[np.argmin(np.abs(xnew-1010e-6))]
print c10um


rb = 2.5e-6
e0 = 9e-12
er = 5
chi = er - 1

alpha_th = 4./3.*np.pi*(rb)**3*e0*chi*(3./(er + 2.))

distst = np.load('distdata.npy')

plt.plot((xnew[:-2]-0.001)*1e6, -coupling*alpha_th*25, linewidth = 5, label = 'Expected force from 5V')
plt.plot(distst[0]+2.5, distst[1], 'o', label = "Measured Force from 5V")
plt.legend()
plt.xlabel("Distance[um]")
plt.ylabel("Dipole force [N]")
plt.show()
