import numpy as np
import seismic_noise as sn
import matplotlib.pyplot as plt
import sys



freqs = np.load('freqs.npy')
ws = 2.*np.pi*freqs

tf = lambda w: sn.hotf(w, 2.*np.pi*100, 1e-13, 1000)
tabtf = lambda w: sn.tabletf(w, 2.*np.pi*100, 1e-13, 1000)

k = (2.*np.pi*100)**2*1e-13
w0 = 2.*np.pi*100

fig = plt.figure()

plt.loglog(freqs, np.abs(tabtf(ws)), linewidth = 2)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Table motion to apparent force transfer [N/M]')
fig.set_size_inches(5.5, 4.5)
#plt.xlim([10, 1000])
plt.subplots_adjust(bottom = 0.15, top = 0.95, left = 0.18, right = 0.95)
plt.savefig('tabtrans.pdf')
plt.show()

fig = plt.figure()
plt.loglog(freqs, np.abs(sn.sql(ws, 2.5)/tf(ws)), linewidth = 2)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Standard quantum limit [N/$\sqrt{Hz}$]')
fig.set_size_inches(5.5, 4.5)
#plt.xlim([10, 1000])
plt.subplots_adjust(bottom = 0.15, top = 0.95, left = 0.18, right = 0.95)
plt.savefig('sql.pdf')
plt.show()

fig = plt.figure()
plt.loglog(freqs, ws*(np.abs(sn.sql(ws, 2.5)/tf(ws)))/np.abs(tabtf(ws)), linewidth = 2)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Table stability requirement [m/s/$\sqrt{Hz}$]')
fig.set_size_inches(5.5, 4.5)
#plt.xlim([10, 1000])
plt.subplots_adjust(bottom = 0.15, top = 0.95, left = 0.18, right = 0.95)
plt.savefig('table_req.pdf')
plt.show()

dat161 = np.load('dat161.npy')
dates = np.load('dates.npy')
datbd = np.load('datbd.npy')
freqs = np.load('freqs.npy')


fig = plt.figure()
plt.loglog(freqs, np.sqrt(dat161[1]), linewidth = 2, label = 'room 161 floor')
plt.loglog(freqs, np.sqrt(dates[1]), linewidth = 2, label = 'end station floor')
#plt.loglog(freqs, np.sqrt(datbd[0]), linewidth = 2, label = 'current optics table')
#plt.loglog(freqs, ws*(np.abs(sn.sql(ws, 2.5)/tf(ws)))/np.abs(tabtf(ws)), linewidth = 2, label = 'table top requirement')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Surface motion [m/s/$\sqrt{Hz}$]')
plt.xlim([10, 1000])
plt.legend()
fig.set_size_inches(11, 8.5)
plt.subplots_adjust(bottom = 0.15, top = 0.95, left = 0.18, right = 0.95)
plt.savefig('bid_asds.pdf')
plt.show()

ws = np.pi*2.*freqs

f161 = np.sqrt(dat161[1]).T*np.abs(tabtf(ws))/ws
fes = np.sqrt(dates[1]).T*np.abs(tabtf(ws))/ws
ftab = np.sqrt(datbd[0]).T*np.abs(tabtf(ws))/ws
fig = plt.figure()
plt.loglog(freqs, f161[0], linewidth = 2, label = 'room 161 floor')
plt.loglog(freqs, fes[0], linewidth = 2, label = 'end station floor')
#plt.loglog(freqs, ftab[0], label = 'current optics tabel')
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Limited force sensitivity [N/$\sqrt{Hz}$]')
plt.xlim([10, 1000])
fig.set_size_inches(11, 8.5)
plt.subplots_adjust(bottom = 0.15, top = 0.95, left = 0.18, right = 0.95)
plt.savefig('vibfs.pdf')
plt.show()

opt = lambda w: sn.hotf(w, 2.*np.pi*1, 1e3, 1)

k = 1e3*(2.*np.pi*1)**2

tfvec = np.abs(opt(ws))*k

fig = plt.figure()
#plt.loglog(freqs, np.sqrt(dat161[1]), linewidth = 2, label = 'room 161 floor')
#plt.loglog(freqs, np.sqrt(dates[1]), linewidth = 2, label = 'end station floor')
plt.loglog(freqs, np.sqrt(dat161[1]).T[0]*tfvec, 'b--',linewidth = 1, label = 'room 161 optics attenuated')
plt.loglog(freqs, np.sqrt(dates[1]).T[0]*tfvec, 'g--',linewidth = 1,label = 'end station floor attenuated')
#plt.loglog(freqs, np.sqrt(datbd[0]), linewidth = 2, label = 'current optics table')
plt.loglog(freqs, ws*(np.abs(sn.sql(ws, 2.5)/tf(ws)))/np.abs(tabtf(ws)),'r', linewidth = 2, label = 'table top requirement')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Surface motion [m/s/$\sqrt{Hz}$]')
plt.xlim([10, 1000])
plt.legend()
fig.set_size_inches(5.5, 4.5)
plt.subplots_adjust(bottom = 0.15, top = 0.95, left = 0.18, right = 0.95)
plt.savefig('attenuated.pdf')
plt.show()
