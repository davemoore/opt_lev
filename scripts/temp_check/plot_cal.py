import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import glob
import matplotlib.mlab as mlab
import scipy.signal as sp
import scipy.optimize as opt
from scipy.special import wofz

dpath = "/Users/dcmoore/Documents/data/"

flist = glob.glob(dpath + "calibration1p_2/*.h5")

flength = 262144
Fs = 10000.

rho = 1800.
m = 4./3 * np.pi * ((10.3e-6)/2.)**3 * rho
mlo = 4./3 * np.pi * ((10.3e-6 - 1.4e-6)/2)**3 * rho
mhi = 4./3 * np.pi * ((10.3e-6 + 1.4e-6)/2)**3 * rho

print(m)

cal_dat_in = np.zeros(flength)
cal_dat_out = np.zeros(flength)
cal_volt = np.zeros(flength)

for f in flist[:1]:

    cdat, a, ff = bu.getdata(f)

    cal_dat_in += cdat[:,0]
    cal_dat_out += cdat[:,4]
    cal_volt += cdat[:,3]

print(len(flist))
#cal_dat_in /= len(flist)
#cal_dat_out /= len(flist)
#cal_volt /= len(flist)

# plt.figure()
# plt.plot(cal_dat_in)
# plt.plot(cal_dat_out)
# plt.plot(cal_volt)
# plt.show()

test, bp, bcov = bu.get_calibration(f, [30,150], make_plot=True, 
                    data_columns = [0,1])

## FFT to get cal frequency

#cpsd, freq = mlab.psd( cal_volt, Fs = Fs, NFFT = flength)

b,a = sp.butter(3, np.array([34.6, 36.6])/(Fs/2), btype="bandpass")
cvf = sp.filtfilt(b,a,cal_volt)
cdif = sp.filtfilt(b,a,cal_dat_in)
cdof = sp.filtfilt(b,a,cal_dat_out)

#cpsdf, freq = mlab.psd( cvf, Fs = Fs, NFFT = flength)

# plt.figure()
# plt.loglog(freq, cpsd)
# plt.loglog(freq, cpsdf)
# plt.show()

gst, gend = 15000, flength-15000
print(np.std( cvf[gst:gend] )*200/0.0029, np.std( cdif[gst:gend] ), np.std( cdof[gst:gend] ) )

force = np.std( cvf[gst:gend] )*200/0.0029 * np.sqrt(2) * bu.e_charge ## force in N
print("volt: ", np.std( cvf[gst:gend] ) * 200 * np.sqrt(2))
print("Force: ", force)
cal_in = np.std( cdif[gst:gend] )*np.sqrt(2)
cal_out = np.std( cdof[gst:gend] )*np.sqrt(2)

print("In loop cal [V/N]: %.2e"%cal_in)
print("Out loop cal [V/N]: %.2e"%cal_out)

#plt.figure()
#plt.plot(cvf)
#plt.plot(cdif)
#plt.plot(cdof)
#plt.show()

## now load the 1mbar data and fit

fmbar = "/Users/dcmoore/Documents/data/1mbar_zcool.h5"
mbardat, at, ff = bu.getdata(fmbar)

# plt.figure()
# plt.plot(mbardat[:,0])
# plt.plot(-mbardat[:,4])
# plt.show()

print(np.shape(mbardat))
cpsdi, freq = mlab.psd( mbardat[:,0], Fs = Fs, NFFT = flength)
cpsdo, freq = mlab.psd( mbardat[:,4], Fs = Fs, NFFT = flength)

#plt.figure()
#plt.loglog(freq, cpsdi)
#plt.loglog(freq, cpsdo)

test, bp, bcov = bu.get_calibration(fmbar, [30,150], make_plot=True, 
                    data_columns = [0,1])

print("Amp: %.2e, %.2e"%(bp[0], np.sqrt(bcov[0,0])))
print("f0: ", bp[1], np.sqrt(bcov[1,1]))

print("amp temp: ", bp[0]*m*np.pi/bu.kb )

cal_in_m = cal_in/(force * m * (2*np.pi*bp[1])**2)
cal_out_m = cal_out/(force * m * (2*np.pi*bp[1])**2)

print("T [K]: ", bp[0]/cal_in_m * m/(2*1.38e-23) )

b,a = sp.butter(3, np.array([25, 1000])/(Fs/2), btype="bandpass")
cvf2 = sp.filtfilt(b,a,mbardat[:,0])

a1mbar = np.std( cvf2[gst:gend] )*np.sqrt(2)

temp = 1./(bu.kb) * (force)**2/(m * (2*np.pi*bp[1])**2) * (a1mbar/cal_in)**2
tlo = 1./(bu.kb) * (force)**2/(mhi * (2*np.pi*bp[1])**2) * (a1mbar/cal_in)**2
thi = 1./(bu.kb) * (force)**2/(mlo * (2*np.pi*bp[1])**2) * (a1mbar/cal_in)**2

err = (thi-tlo)/2

print( "Temp: %.1f +/- %.1f"%(temp,err)  )

#print( (a1mbar/cal_in)**0.5 * force/(2*np.pi*bp[1])**2 * 1/(bu.kb) )

cal_fac = (1.7/1.1*force/(m*((2*np.pi*bp[1])**2)))/(np.std(cdif[gst:gend])*np.sqrt(2))

print(bp[1])
print("cal fac: ", (cal_fac))

print(np.std(cdif[gst:gend])*np.sqrt(2))
print(np.std(cvf2[gst:gend])*np.sqrt(2))

plt.figure()
plt.plot( cdif * cal_fac )
plt.plot( cvf2 * cal_fac )

tvec = np.arange(0,flength/Fs,1./Fs)

temp = 1/tvec[-1] * np.trapz(( cvf2 * cal_fac )**2, tvec) * m * (2*np.pi*bp[1])**2/bu.kb * 1./(2*np.pi)

cal_fac = (1.7/1.1*force/(mlo*((2*np.pi*bp[1])**2)))/(np.std(cdif[gst:gend])*np.sqrt(2))
temp_hi = 1/tvec[-1] * np.trapz(( cvf2 * cal_fac )**2, tvec) * mlo * (2*np.pi*bp[1])**2/bu.kb * 1./(2*np.pi)

cal_fac = (1.7/1.1*force/(mhi*((2*np.pi*bp[1])**2)))/(np.std(cdif[gst:gend])*np.sqrt(2))
temp_lo = 1/tvec[-1] * np.trapz(( cvf2 * cal_fac )**2, tvec) * mhi * (2*np.pi*bp[1])**2/bu.kb * 1./(2*np.pi)

print("tempss: ", temp, (temp_hi-temp_lo)/2)
print( 2.11e-13/(2*np.pi) * m * (2*np.pi*bp[1])**2/bu.kb)
plt.close('all')
#plt.show()

## first get a good file that we can subtract and get the noise lines

colors = bu.get_color_map(18)
dg_vec = []
idx_vec = []
for f in range(1,18):
    path = dpath + "temp_x9/%d/*.h5"%f
    files = glob.glob(path)
    for fi in files:
        cdat, a, ff = bu.getdata(fi)
        dg = a['PID'][0]
        break
    dg_vec.append(dg)
    idx_vec.append(f)

dg_vec, idx_vec = zip(*sorted(zip(dg_vec, idx_vec)))

def ffn(f,A,f0,gam):
    omega = 2*np.pi*f
    omega_0 = 2*np.pi*f0
    return np.sqrt( np.abs(A)*gam/((omega_0**2 - omega**2)**2 + omega**2*gam**2) )

def ffn2(f,A,f0,gam,sig):
    omega = 2*np.pi*f
    omega_0 = 2*np.pi*f0
    z = ((omega**2 - omega_0**2) + 1j * omega*gam)/(np.sqrt(2)*sig)
    V = np.abs(A*np.real( wofz(z) )/sig)
    return np.sqrt( V )
    
bad_ranges = [[42.8, 45.6],
              [50, 51],
              [54.4, 55.3],
              [56.8, 57.5],
              [59.0, 60.4],
              [65.4, 65.8],
              [70.8, 71.3],
              [72.1, 73.3],
              [40,41],
              [36.5,38.5],
              [74.6, 75.5],
              [78.9,80.15],
              [80.65, 81.6],
              [86.0, 86.8],
              [88.5, 90.3],
              [94.7, 95.8]]

flength = 2**16
## make no sphere data
path = dpath + "nosphere/*.h5"

files = glob.glob(path)

psd_in = []
psd_out = []
nf=0
for fi in files:

    cdat, a, ff = bu.getdata(fi)
    dg = a['PID'][0]

    #print(np.shape(cdat))
    cpsd1, freq = mlab.psd( cdat[:,0]*cal_fac, Fs = Fs, NFFT = flength)
    cpsd2, freq = mlab.psd( cdat[:,4]*cal_fac, Fs = Fs, NFFT = flength)

    if len(psd_in)==0:
        psd_in = 1.0*cpsd1
        psd_out = 1.0*cpsd2
    else:
        psd_in += cpsd1
        psd_out += cpsd2
    nf+=1

psd_in /= nf
psd_out /= nf

noise_in = 1.0*psd_in
noise_out = 1.0*psd_out

## now loop through all the files and plot the temps
fig1=plt.figure()
fig2=plt.figure()

import matplotlib.gridspec as gridspec
fig3 = plt.figure()
gs1 = gridspec.GridSpec(6, 3)
gs1.update(wspace=0, hspace=0)

spars = [2.42100889e-12, 6.32709456e+01, 1.79414730e-02, 6.99109858e+02]
pltidx = 0
temp_pts = []
for f in idx_vec[::-1]:
    path = dpath + "temp_x9/%d/*.h5"%f

    files = glob.glob(path)

    psd_in = []
    psd_out = []
    nf=0
    for fi in files:

        cdat, a, ff = bu.getdata(fi)
        dg = a['PID'][0]
        
        #print(np.shape(cdat))
        cpsd1, freq = mlab.psd( cdat[:,0]*cal_fac, Fs = Fs, NFFT = flength)
        cpsd2, freq = mlab.psd( cdat[:,4]*cal_fac, Fs = Fs, NFFT = flength)

        if len(psd_in)==0:
            psd_in = 1.0*cpsd1
            psd_out = 1.0*cpsd2
        else:
            psd_in += cpsd1
            psd_out += cpsd2
        nf+=1

    psd_in /= nf
    psd_out /= nf

    psd_in -= noise_in
    psd_out -= noise_out

    psd_in = np.abs(psd_in)
    psd_out = np.abs(psd_out)
    
    ccol = colors[np.argwhere( dg_vec == dg )[0][0]]

    if pltidx <5:
        gpts = np.logical_and( freq>=56., freq<=70.)
    elif pltidx <7:
        gpts = np.logical_and( freq>=50., freq<=80.)
    elif pltidx <10:
        gpts = np.logical_and( freq>=35., freq<=99.)
    else:
        gpts = np.logical_and( freq>=35., freq<=120.)
        gpts = np.logical_and( gpts, np.logical_or( freq<83.0, freq>98 ) )
        
    fpts = gpts
    for b in bad_ranges:
        fpts = np.logical_and( fpts, np.logical_or( freq<b[0], freq>b[1] ) )

    if pltidx >= 2:
        fpts = np.logical_and( fpts, np.logical_or( freq<63.2, freq>63.8 ) )

    if pltidx < 2:
        fitfun = ffn2
    elif pltidx ==2:
        fitfun = ffn
        spars = spars[:3]
    elif pltidx <=8:
        fitfun = ffn
    elif pltidx ==9:
        fitfun = lambda x,A,gam: ffn(x,A,68.5,gam)
        spars = [spars[0], spars[2]]
    elif pltidx <12:
        fitfun = lambda x,A,gam: ffn(x,A,68.5,gam)
    elif pltidx ==12:
        fitfun = lambda x,A: ffn(x,A,68.5,1000)  ## essentially flat so need to fix gam as well
        spars = [spars[0],]
    else:
        fitfun = lambda x,A: ffn(x,A,68.5,1000)
        
    if pltidx < 10:
        errs = np.maximum(  0.1*np.sqrt(psd_in[fpts]), np.median(noise_in) )
    else:
        errs = np.ones_like( freq[fpts] )
        
    bp, bcov = opt.curve_fit(fitfun, freq[fpts], np.sqrt(psd_in[fpts]), p0=spars, sigma = errs)
    spars = bp
    print(bp)

    if pltidx < 10:
        errs = np.maximum(  0.1*np.sqrt(psd_out[fpts]), np.median(noise_out) )
    else:
        errs = np.ones_like( freq[fpts] )
        
    bp2, bcov2 = opt.curve_fit(fitfun, freq[fpts], np.sqrt(psd_out[fpts]), p0=spars, sigma = errs)
    
    
    xxint = np.linspace(10,1e3, 1e4)
    fint = np.trapz( fitfun(xxint,*bp)**2, xxint )
    fint2 = np.trapz( fitfun(xxint,*bp2)**2, xxint )
    if pltidx<9:
        temp_pts.append( [dg, fint/(2*np.pi) * m * (2*np.pi*bp[1])**2/bu.kb, fint2/(2*np.pi) * m * (2*np.pi*bp[1])**2/bu.kb] )
    else:
        temp_pts.append( [dg, fint/(2*np.pi) * m * (2*np.pi*68.5)**2/bu.kb, fint2/(2*np.pi) * m * (2*np.pi*68.5)**2/bu.kb] )

    xx = np.linspace(56, 70, 1000)
    if(False):
        xxp = np.linspace(10, 500, 10000)
        plt.close('all')
        plt.figure()
        plt.semilogy(freq[fpts], np.sqrt(psd_in[fpts]), '.', color=ccol)
        plt.plot( xxp, fitfun(xxp, *bp), color=ccol)
        plt.xlim([5,500])
        plt.title(str(pltidx))
        plt.show()
    
    plt.figure(fig1.number)
    plt.semilogy(freq[gpts], np.sqrt(psd_in[gpts]), '.', color=ccol, label=str(f)) ##dg))
    plt.plot( xx, fitfun(xx, *bp), color=ccol)
    plt.xlim([56,70])
    plt.figure(fig2.number)
    plt.semilogy(freq[gpts], np.sqrt(psd_out[gpts]), '.', color=ccol, label=str(dg))
    plt.plot( xx, fitfun(xx, *bp2), color=ccol)
    plt.xlim([56,70])

    plt.figure(fig3.number)
    ax1 = plt.subplot(gs1[pltidx])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.semilogy(freq[gpts], np.sqrt(psd_in[gpts]), '.', color=ccol, label=str(f)) ##dg))
    plt.plot( xx, fitfun(xx, *bp), color=ccol)

    pltidx += 1
    
plt.figure(fig1.number)
#plt.legend()
plt.title("in loop")
plt.savefig("inloop_all.pdf")
plt.figure(fig2.number)
plt.title("out loop")
plt.savefig("outloop_all.pdf")
#plt.legend()

plt.figure(fig3.number)
plt.title("in loop")
plt.savefig("inloop_fits.pdf")


plt.figure()
temp_pts = np.array(temp_pts)
plt.loglog( np.abs(temp_pts[:,0]), temp_pts[:,1]*1e6, 'bo')
plt.loglog( np.abs(temp_pts[:,0]), temp_pts[:,2]*1e6, 'ro')
plt.xlabel("abs(dg)")
plt.ylabel("Temp [uk]")
plt.savefig("temp.pdf")


plt.show()
