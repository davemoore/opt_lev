#!/usr/bin/env python

import numpy
from pylab import *

l,g,x,d,s=numpy.loadtxt("times.txt",unpack=True,skiprows=1)

gs=numpy.unique(g)
ls=numpy.unique(l)
xs=numpy.unique(x)
vals=ls*0
for i in range(0,len(gs)):
    for ll in range(0,len(ls)):
        failed=where((g == gs[i]) & (s==0) & (l == ls[ll]))
        if(len(failed[0]) > 0):
            print(gs[i],ls[ll],len(failed[0]))

for i in range(0,len(gs)):
    vals=ls*0
    for ll in range(0,len(ls)):
        inds = numpy.where((g == gs[i]) & (s == 1) & (l==ls[ll]))
        if(len(inds[0]) > 1):
            vals[ll]=sum(d[inds])
    if(sum(vals) > 1000):
        inds = where(vals == 0)
        vals[inds]=max(vals)
        plot(ls,vals/(3600*24),label=str(gs[i]))

xscale('log')
yscale('log')
xlabel('L')
ylabel('Time (days)')
legend(loc="lower right",title="Grid")
savefig('timevseparation.png')

clf()
for ll in range(0,len(ls)):
    vals=gs*0
    for i in range(0,len(gs)):
        inds = numpy.where((g == gs[i]) & (s == 1) & (l==ls[ll]))
        if(len(inds[0]) > 1):
            vals[i]=sum(d[inds])
    if(sum(vals) > 1000):
        #inds = where(vals == 0)
        #vals[inds]=max(vals)
        plot(gs,vals/(3600*24),label=str(ls[ll]))

xscale('log')
yscale('log')
xlabel('Gridding')
ylabel('Time (days)')
legend(loc="lower right",title="L")
savefig('TimevGridding.png')

clf()
gs=numpy.unique(g)
vals=gs*0
for i in range(0,len(gs)):
    inds = numpy.where((g == gs[i]) & (s == 1))
    if(len(inds[0]) > 1):
        vals[ll]=max(d[inds])

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
