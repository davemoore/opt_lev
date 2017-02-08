## load a file containing values of the voltage vs beta and remake it, assuming
## that the force scales like the square of the voltage

import numpy as np
import matplotlib.pyplot as plt

#freqs = [5,7,9,13,17,23,29,37,43,51,57,69,71,111,]
#freqs = np.linspace(5,150,10)


Fsamp = 5000.
cutsamp = 2000
Npoints = 250000
drive_elec = 5
drive_voltage = 1

scaled_amp = 4  # max voltage played by DAC

#drive_freqs = np.linspace(1, 200, 100)
#drive_freqs = np.linspace(1, 3, 5)
drive_freqs = np.linspace(1, 200, 100)


#np.random.seed(123)
np.random.seed()
random_phase = True
optimize_phase = False #True
load_opt_phase = True #False

passes = 1000

phase_fil = "opt_phases_higherf.txt"
out_phase_fil = "opt_phases_higherf.txt"

out_script = r'C:\GitHub\opt_lev\labview\DAQ_settings\freq_comb_elec%i_1-500Hz_%iVppamp.txt' % (drive_elec, int(scaled_amp))

######################################


dt = 1. / Fsamp
t = np.linspace(0, (Npoints-1) * dt, Npoints)

fft_freqs = np.fft.rfftfreq(Npoints - cutsamp, dt)

for i in range(len(drive_freqs)):
    ind = np.argmin(np.abs(fft_freqs - drive_freqs[i]))
    drive_freqs[i] = fft_freqs[ind]

drive_arr = np.zeros(Npoints)

if optimize_phase and not load_opt_phase:
    print "optimizing phases..."
    scales = []
    phase_arrs = []
    k = 0
    for i in range(passes):
        if k <= (100 * float(i) / passes):
            print k,
            k += 1
        comb = np.zeros(Npoints)
        phases = []
        for freq in drive_freqs:
            phase = np.random.random() * 2 * np.pi
            comb += drive_voltage*np.sin(2 * np.pi * freq * t + phase)
            phases.append(phase)
        phases = np.array(phases)
        phase_arrs.append(phases)

        scale = scaled_amp / np.max(np.abs(comb))
        
        scales.append(scale)

    plt.plot(scales)
    plt.show()
    
    ind = np.argmax(scales)

    phases = phase_arrs[ind]
    opt_phase_out = np.column_stack((drive_freqs, phases))
    
    np.savetxt(out_phase_fil, opt_phase_out)

    for i in range(len(drive_freqs)):
        freq = drive_freqs[i]
        phase = phases[i]
        drive_arr += drive_voltage*np.sin(2 * np.pi * freq * t + phase)


elif load_opt_phase:
    data = np.loadtxt(phase_fil)
    freqs = data[:,0]
    phases = data[:,1]
    for i in range(len(freqs)):
        freq = freqs[i]
        phase = phases[i]
        drive_arr += drive_voltage*np.sin(2 * np.pi * freq * t + phase)

        
else:
    for freq in drive_freqs:
        phi = 0
        if random_phase:
            phi += np.random.random() * 2 * np.pi
        drive_arr += drive_voltage * np.sin(2 * np.pi * freq * t + phi)


        
scale = scaled_amp / np.max(drive_arr)
    
drive_arr = drive_arr * scale
    
ft = np.fft.rfft(drive_arr[cutsamp:])
ft_freqs = np.fft.rfftfreq(len(drive_arr[cutsamp:]),dt)

plt.figure()
plt.plot(t, drive_arr)
plt.title("Component Amplitude %.2e"%scale)

plt.figure()
plt.loglog(ft_freqs, (ft * ft.conj()).real)

plt.show()

out_arr = []
for ind in range(8):
    if ind == drive_elec:
        out_arr.append(drive_arr)
    else:
        out_arr.append(np.zeros(Npoints))

out_arr = np.array(out_arr)
#print out_arr.shape

#np.savetxt(r'C:\GitHub\opt_lev\labview\DAQ_settings\freq_comb_elec6_optphase2.txt', out_arr, fmt='%.5e', delimiter=",")

np.savetxt(out_script, out_arr, fmt='%.5e', delimiter=",")
    
