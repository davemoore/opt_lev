import numpy as np
import matplotlib.pyplot as plt
import os, re, glob
import bead_util as bu
import scipy.optimize as opt
import cPickle as pickle
import opt_filt_util as ofu
import sys
import matplotlib

#path to directory with chirps and charged bead
path = "/data/20140623/Bead1/one_charge_chirp"

## path to save plots and processed files (make it if it doesn't exist)
outpath = "/home/arider/analysis" + path[5:]
if( not os.path.isdir( outpath ) ):
    os.makedirs(outpath)

    

H_path = "/home/arider/analysis/20140623/Bead1/one_charge_chirp"
noise_path = "/data/20140623/Bead1/chirp_plates_terminated"

recalc_noise = False
reprocess_file = True


freqs = np.load(os.path.join(H_path, 'Hfreqs.npy'))
H = np.load(os.path.join(H_path, 'H.npy'))

if reprocess_file:
    init_list = glob.glob(path + "/*.h5")
    files = sorted(init_list, key = bu.find_str)

    #files = files[::10]

    As = np.zeros(len(files))
    Rcs = np.zeros(len(files))
    As1 = np.zeros(len(files))
    Rcs1 = np.zeros(len(files))


    N_pts = len(freqs)

    curr_dict = ofu.getdata_fft(files[0], path)
    all_freqs = np.fft.rfftfreq(curr_dict["N"], 1./curr_dict['fsamp'])
    bfreqs = np.array([i in freqs for i in all_freqs])
    print np.sum(bfreqs)
    if recalc_noise:
        noise = ofu.get_noise(noise_path)
    else:
        noise = np.load("/home/arider/analysis/20140623/Bead1/plates_terminated/noise.npy")

    #plt.loglog(noise)
    #plt.show()

    #noise = noise*0 + np.mean(noise) 
    for i in range(len(files)):
    
            try:
                curr_dict = ofu.getdata_fft(files[i], path)
                
                dfft = curr_dict['drive_fft'][bfreqs]
                rfft = curr_dict['response_fft'][bfreqs]
                noisei = noise[bfreqs]

                f = lambda A: ofu.freq_fit(A, H, dfft, rfft, noisei)

                res = opt.minimize_scalar(f)
                #print res
                As[i] = res.x
                Rcs[i] = res.fun/(N_pts-2)
                
                Ahat, sig_Ahat, rcs = ofu.opt_filter(dfft, rfft, noisei, H)
                As1[i] = Ahat
                Rcs1[i] = rcs
                print curr_dict["N"]
        
            except:
                print "Holy Shit:",sys.exc_info()[0]


                print 'lootp finished'

    np.save(os.path.join(outpath, 'As'), As)
    np.save(os.path.join(outpath, 'Rcs'), Rcs)

else:
    As = np.load(os.path.join(outpath, 'As.npy'))
    Rcs = np.load(os.path.join(outpath, 'Rcs.npy'))

As1 = np.real(As1)

print np.mean(As1)
print np.std(As1)/np.sqrt(len(As1)-1)
print sig_Ahat

fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(As, 'r.')
plt.plot(np.real(As1), 'b.')
plt.ylabel('Charge')
plt.subplot(2, 1, 2)
plt.plot(Rcs, 'r.')
plt.plot(Rcs1, 'b.')
plt.ylabel('Reduced Chi-square score')
plt.xlabel("Frequency[Hz]")
plt.show()
