#!/usr/bin/env python

import numpy
from pylab import *

l,g,x,d,s=numpy.loadtxt("times.txt",unpack=True,skiprows=1)

gs=numpy.unique(g)
vals=gs*0
for i in range(0,len(gs)):
    inds = numpy.where((g == gs[i]) & (s == 1))
    if(len(inds[0]) > 1):
        vals[i]=max(d[inds])

plot(gs,vals,label="Gridding")

ls=numpy.unique(l)
vals=ls*0
for i in range(0,len(ls)):
    inds = numpy.where((l == ls[i]) & (s == 1))
    if(len(inds[0]) > 1):
        vals[i]=max(d[inds])

plot(ls,vals,label="Separation")

xs=numpy.unique(x)
vals=xs*0
for i in range(0,len(xs)):
    inds = numpy.where((x == xs[i]) & (s == 1))
    if(len(inds[0]) > 1):
        vals[i]=max(d[inds])

plot(xs,vals,label="Integrand Argument")

xscale('log')
yscale('log')
xlabel('Value')
ylabel('Convergence Time (s)')
legend(loc='lower left')

savefig('times.png')
