## plot the theoretical force sensitivity vs pressure, 
## see Li thesis, eqn 6.14

import numpy as np
import matplotlib.pyplot as plt

p = np.logspace(-10, 3, 1e2) ## in mbar
bead_rad = 2.3e-6 ## m
bead_dens = 2e3 ## kg/m^3
air_viscosity = 18.54e-6 ## Pa s
kb = 1.38e-23
T = 300.
gas_diam = 1.42e-10 ## m

bead_mass = 4./3*np.pi*bead_rad**3 * bead_dens

p_pa = p*100. ##pascal

l = 101325/p_pa * 68e-9 ## scaling mfp to 68 nm at 1 atm, from wikipedia
kn = l/bead_rad
ck = 0.31*kn/(0.785 + 1.152*kn + kn**2)

gamma = 6*np.pi*air_viscosity*bead_rad/bead_mass * 0.619/(0.619 + kn) * (1+ck)

## force sensitivity/sqrt(Hz) from sqrt( 4*m*Gamma*kT)

sig_f = np.sqrt(4*bead_mass*gamma*kb*T)

fig = plt.figure()
plt.loglog( p, sig_f)
plt.xlabel("Pressure [mbar]")
plt.ylabel(r"Force sensitivity, $\sigma_F$ [N Hz$^{-1/2}$]")
plt.grid("on")
fig.set_size_inches(8,6)
plt.savefig("f_v_p.pdf")
plt.show()
