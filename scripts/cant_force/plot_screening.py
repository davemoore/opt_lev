## upper limit on beta from vaccum chamber reaching eq pot
import numpy as np
import matplotlib.pyplot as plt

lam_list = [0.1, 0.23, 1., 2.3, 10, 23, 100]

hbarc = 2e-16 ## GeV m
rv = 2e-3/hbarc ## GeV^-1
mpl = 2e18 ## GeV
rho_vac = 4.7e-30 ## GeV^4

bb = np.logspace(0,12,1e2)

for lam in lam_list:
    
    lam_gev = lam * 1e-12

    phi_vac = 0.55*(2. * lam_gev**5 * rv**2)**(1./3) * 1/hbarc ## m
    phi_eq = (mpl*lam_gev**5/(bb*rho_vac))**(1./2) * 1/hbarc ## m


    plt.figure()
    plt.loglog( bb, phi_eq ) 
    plt.loglog( bb, phi_vac*np.ones_like(bb), 'k--' )
    plt.title(str(lam))

    plt.show()
