import cant_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import Tkinter
import tkFileDialog
import os, sys
from scipy.optimize import curve_fit
import bead_util as bu
from scipy.optimize import minimize_scalar as minimize 

dirs = [9,]
cal = 5.0e-14

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )

load_from_file = False

pos = []

def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-2]])
    dir_obj.load_dir(cu.simple_loader)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)

xpos = []
ypos = []
zpos = []


for obj in dir_objs:
    for fobj in obj.fobjs:
        xpos.append(fobj.get_stage_settings(axis=0)[0])
        ypos.append(fobj.get_stage_settings(axis=1)[0])
        zpos.append(fobj.get_stage_settings(axis=2)[0])

print np.unique(xpos)
print np.unique(ypos)
print np.unique(zpos)

