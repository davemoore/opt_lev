import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt


n = 2**14
t = np.linspace(0, 100,n)

w = 1.
wmod = 3.2
sig = ss.sawtooth(2.*np.pi*w*t, width = 1/2.)

plt.plot(t, sig)
plt.show()

mod = np.sin(2.*np.pi*wmod*t)

fft = np.fft.rfft(sig)
freqs = np.fft.rfftfreq(n)*n/100.

fn = w
sfreqs = []
while fn <= np.max(freqs):
    sfreqs += [np.argmin(np.abs(freqs - fn))]
    fn += fn

fft2 = np.zeros(len(fft))

for i in range(len(freqs)):
    if i in sfreqs:
        fft2[i] = fft[i]

 
sig2  = np.fft.irfft(fft2)

plt.loglog(freqs, np.abs(fft))
plt.loglog(freqs[sfreqs],np.abs(fft[sfreqs]), 'xr' )
plt.show()

plt.plot(t, sig2)
plt.show()
