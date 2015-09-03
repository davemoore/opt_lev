import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


wfm_file = '/home/arider/opt_lev/waveforms/rand_wf_primes.txt'

def  ho_transfunc(freqs, f0, Q):
    #returns a harmonic oscillator transfeer function with resonant frequency fo and quality factor Q
    w0 = 2.*np.pi*f0
    sci = 1./(2.*Q)
    ws = 2.*np.pi*freqs
    return 1./(-ws**2 + w0**2 + 2.j*sci*w0*ws)
  

def wfm_realization(wfm_tmp, n, Fs):
    #Repeats wfm_tmp n times and samples it at the points in t.Plays wfm_tmp bact at 1Hz
    t = np.arange(0, 1., 1./Fs)
    wfm_t = np.arange(0, 1., 1./len(wfm_tmp))
    wfm_r_spline = UnivariateSpline(wfm_t, wfm_tmp)
    samped_wfm = wfm_r_spline(t)
    wfm_r = samped_wfm
    for i in range(int(n-1)):
        wfm_r = np.append(wfm_r, samped_wfm)
    
    return wfm_r/np.max(wfm_r) 

def noise_realization(psd_amp, T, Fs):
    #Generates an fft of a noise rtealization from a noise power.
    nfft = int(T*Fs/2) + 1
    return np.sqrt(psd_amp*T*Fs**2/(4.))*(np.random.randn(nfft) + 1.j*np.random.randn(nfft))

T = 100.
Fs = 5000.
t = np.arange(0, T, 1./Fs)
freqs = np.fft.rfftfreq(len(t), 1./Fs)
wfm = np.loadtxt(wfm_file)

samped_wfm = wfm_realization(wfm, T, Fs)

wfm_fft = np.fft.rfft(samped_wfm)

n_sims = 100
As = np.zeros(n_sims)
inds = range(n_sims)

drive_amp = 3.
noise_amp = 1.

H = 1.#ho_transfunc(freqs, 100, 10)

for i in inds:
    noise = noise_realization(noise_amp, T, Fs)
    rec_fft = (drive_amp*wfm_fft + noise)*H
    As[i] = np.real(bu.amp_opt_filter(rec_fft, freqs*0. + noise_amp, wfm_fft*H))
    
    
plt.plot(As)
plt.show()

print 'variance should be:', bu.amp_opt_filter_var(wfm_fft*H, np.ones(len(freqs))*noise_amp, T, Fs)

print 'variance is:', np.std(As)**2


