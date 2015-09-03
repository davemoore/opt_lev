import numpy as np
import scipy.signal as sig

def get_primes(max_val):
#find all prime numbers less than max value.  
  seive = range(2, max_val + 1)
  for x in seive:
    if x in seive:
      y = 2*x
      while y <= max_val:
        if y in seive:
          seive.remove(y)
        y = y + x

  return np.array(seive)

def prime_sweep(freqs, phases):
  all_freqs = np.arange(8001)
  fft = np.zeros(8001, dtype=complex)
  for f, i in zip(freqs, range(len(freqs))):
    ind = np.where(all_freqs == f)
    fft[ind] = np.exp(1.j*phases[i])
  return np.fft.irfft(fft)


