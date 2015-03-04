import numpy as np

half_length = 8192
max_val = 2**12-1

xvals = np.arange(-half_length/2,half_length/2)
yvals = np.arange(-half_length/2,half_length/2)

xtot = np.hstack( (xvals, xvals[::-1]) )
ytot = np.hstack( (yvals, yvals[::-1]) )

dtot = np.transpose( np.vstack( (xtot, ytot) ) )

np.savetxt(r"D:\GitHub\opt_lev\labview\fpga\triangle_buffer.txt", dtot, delimiter=",",fmt="%d")

## square
xvals = np.arange(-half_length/4,half_length/4)
yvals = np.arange(-half_length/4,half_length/4)

xtot = np.hstack( (xvals, xvals[-1]*np.ones(half_length/2),
                   xvals[::-1], xvals[0]*np.ones(half_length/2)) )
ytot = np.hstack( (yvals[0]*np.ones(half_length/2),yvals,
                   yvals[-1]*np.ones(half_length/2),yvals[::-1]) )

dtot = np.transpose( np.vstack( (xtot*0.2, ytot) ) )
dtot = 1.0*max_val*dtot/np.max(dtot)
np.savetxt(r"D:\GitHub\opt_lev\labview\fpga\square_buffer.txt", dtot, delimiter=",",fmt="%d")

## circle
t = np.linspace(0,2*np.pi,half_length*2)
xtot = np.round(half_length*np.sin(t))
ytot = np.round(half_length*np.cos(t))

dtot = np.transpose( np.vstack( (xtot, ytot) ) )

dtot = 1.0*max_val*dtot/np.max(dtot)
np.savetxt(r"D:\GitHub\opt_lev\labview\fpga\circle_buffer.txt", dtot, delimiter=",",fmt="%d")

## lissajous
t = np.linspace(0,2*np.pi,half_length*2)
xtot = np.round(half_length*np.sin(1*t))
ytot = np.round(half_length*np.cos(10*t))

dtot = np.transpose( np.vstack( (xtot*0.2, ytot) ) )

dtot = 1.0*max_val*dtot/np.max(dtot)
np.savetxt(r"D:\GitHub\opt_lev\labview\fpga\lissajous_buffer.txt", dtot, delimiter=",",fmt="%d")

## two_traps
t = np.linspace(0,2*np.pi*half_length/2.,num = half_length*2)
xtot = np.abs(np.round(half_length*np.sin(1*t)))
ytot = np.round(half_length*0.*t)

dtot = np.transpose( np.vstack( (xtot, ytot) ) )

dtot = 1.0*max_val*dtot/np.max(dtot)
np.savetxt(r"D:\GitHub\opt_lev\labview\fpga\two_traps.txt", dtot, delimiter=",",fmt="%d")
