import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

niters = 200

npts = 2**21
Fs = 1000.
flist = range(31,130,2)

w0 = 2*np.pi*80.
Gam = 0.1

np.random.seed(4943)

t = np.linspace(0, (npts-1)/Fs, npts)

s = np.zeros_like(t)
for f in flist:
    phi = np.random.rand()*2*np.pi
    A = 1e11/( ((2*np.pi*f)**2 - w0**2)**2 + (2*2*np.pi*f*w0*Gam)**2 )
    phif = np.arctan2( (2*2*np.pi*f*w0*Gam), ((2*np.pi*f)**2 - w0**2) )
    s += A*np.sin( 2*np.pi*f*t + phi + phif )

sf = np.fft.rfft( s )
norm = np.sum( np.abs(sf)**2 )

ps, f = mlab.psd( s, Fs = Fs, NFFT=npts )

fpts_list = []
for ff in flist:
    fpts_list.append( np.argmin( np.abs( f - ff ) ) )

# plt.figure()
# plt.loglog(f,ps)
# plt.show()

out_vec = []
ofavg = np.zeros_like(sf)
for n in range(niters):
    if n % 100 == 0: print(n)
    
    y = np.random.randn( len(t) ) + 0.001*s + 0.1*s**2
    y = y - np.mean(y)
    
    yf = np.fft.rfft( y )

    of = np.conj(sf)*yf/norm

    ofavg += of
    
    out_vec.append( np.sum(of) )

ofavg /= niters
    
hr, ber = np.histogram( np.real( out_vec ), bins=20 )
hi, bei = np.histogram( np.imag( out_vec ), bins=20 )

plt.figure()
plt.step(ber[:-1], hr, where='post', label="Real part, $\mu = %e$"%np.mean(np.real(out_vec)))
plt.step(bei[:-1], hi, where='post', label="Imag part, $\mu = %e$"%np.mean(np.imag(out_vec)))

plt.legend()

oftemp = np.conj(sf)*sf/norm

print(np.sum(np.real(ofavg)), np.sum(np.real(ofavg[fpts_list])))

sfac = np.max( np.abs( np.real(ofavg[fpts_list]) ) )/np.max( np.abs( np.real(oftemp[fpts_list]) ) )
plt.figure()
plt.plot( f[fpts_list], np.real(ofavg[fpts_list]), 'o' )
plt.plot( f[fpts_list], np.imag(ofavg[fpts_list]), 'o' )
plt.plot( f[fpts_list], np.real(oftemp[fpts_list])*sfac )
plt.plot( f[fpts_list], np.imag(oftemp[fpts_list])*sfac )

plt.show()
