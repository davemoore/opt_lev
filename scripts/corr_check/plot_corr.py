import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

niters = 1000

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

fig=plt.figure()
plt.subplot(2,1,1)
plt.semilogy(f,ps)
plt.xlabel('Freq [Hz]')
plt.ylabel('PSD [arb^2/Hz]')
plt.xlim(10,150)
plt.title("Amplitude and phase of template")
plt.subplot(2,1,2)
ang = np.angle(sf)
#ang[ang>0] = ang[ang>0]-2*np.pi
plt.plot(f,ang)
plt.xlim(10,150)
plt.xlabel('Freq [Hz]')
plt.ylabel('Phase [rad]')
fig.set_size_inches(8,6)
plt.tight_layout()
plt.savefig('template.png')

out_vec = []
ofavg = np.zeros_like(sf)
for n in range(niters):
    if n % 100 == 0: print(n)
    
    y = np.random.randn( len(t) ) + 0*s + 0.1*s**2
    y = y - np.mean(y)
    
    yf = np.fft.rfft( y )

    of = np.conj(sf)*yf/norm

    ofavg += of
    
    out_vec.append( np.sum(of) )

ofavg /= niters
    
hr, ber = np.histogram( np.real( out_vec ), bins=20 )
hi, bei = np.histogram( np.imag( out_vec ), bins=20 )


mur, sigr = np.mean(np.real(out_vec)), np.std(np.real(out_vec))/np.sqrt(len(out_vec))
mui, sigi = np.mean(np.imag(out_vec)), np.std(np.imag(out_vec))/np.sqrt(len(out_vec))
fig=plt.figure()
plt.step(ber[:-1], hr, where='post', label="Real part, $\mu = %.1e \pm %.1e$"%(mur,sigr))
plt.step(bei[:-1], hi, where='post', label="Imag part, $\mu = %.1e \pm %.1e$"%(mui,sigi))
plt.legend()
plt.xlabel("Summed correlation [arb]")
plt.ylabel("Counts")
plt.title("Correlation, 1000 iters, 2f signal at 0.1")
fig.set_size_inches(8,6)
plt.tight_layout()
plt.savefig('of_corr_0_1000_with2fsig.png')

oftemp = np.conj(sf)*sf/norm

print(np.sum(np.real(ofavg)), np.sum(np.real(ofavg[fpts_list])))
print(np.sum(np.imag(ofavg)))

sfac = np.max( np.abs( np.real(ofavg[fpts_list]) ) )/np.max( np.abs( np.real(oftemp[fpts_list]) ) )
fig = plt.figure()
plt.plot( f[fpts_list], np.real(ofavg[fpts_list]), 'o', label="Re(data)" )
plt.plot( f[fpts_list], np.imag(ofavg[fpts_list]), 'o', label="Im(data)" )
plt.plot( f[fpts_list], np.real(oftemp[fpts_list])*sfac, label="Re(template)" )
plt.plot( f[fpts_list], np.imag(oftemp[fpts_list])*sfac, label="Im(template)" )
plt.xlabel("Freq [Hz]")
plt.ylabel("Correlation amplitude [arb]")
plt.legend()
plt.title("Correlation shape, 2f signal at 0.1")
fig.set_size_inches(8,6)
plt.tight_layout()
plt.savefig('corr_shape_with_2fsignal.png')

plt.show()
