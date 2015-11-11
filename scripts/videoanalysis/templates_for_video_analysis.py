import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob, os, re
import itertools
import scipy.optimize as opti
import scipy.interpolate as interp
import matplotlib.mlab as mlab

files = []
#vfile = '/data/20151028/bead3/5_3Hz_drive.avi'
vfile1 = '/data/20151029/bead2/cant_sweep_noinputrazor_2_withcam/cant_sweep_noinputrazor_2_withcam.avi'
vfile2 = '/data/20151030/bead1/next_day/ac_bias/ac_bias.avi'

## File name, length of .h5 data files in seconds and number of 
## associated .h5 files (ie. numflashes)
#files.append([vfile1, 50, 5, 'no razer, 3.7 Hz'])
files.append([vfile2, 50, 5, 'razer, 11 Hz'])

n = 0
videodicts = {}
for fil in files:
    new_dict = {}
    new_dict['handle'] = cv2.VideoCapture(fil[0])
    new_dict['fps'] = new_dict['handle'].get(5)
    new_dict['totframes'] = new_dict['handle'].get(7)
    new_dict['h5length'] = fil[1]
    new_dict['h5files'] = fil[2]
    new_dict['label'] = fil[3]
    videodicts[n] = new_dict
    n += 1

realtemplates = []
for i in range(n):
    video = videodicts[i]['handle']
    fps = videodicts[i]['fps']
    totframes = videodicts[i]['totframes']
    file_length = videodicts[i]['h5length']
    num_files = videodicts[i]['h5files']

    ##
    ## Making the bead_template by first finding the bead
    ## with a toy circular template
    ##

    ret, firstframe = video.read()

    gray_frame = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
    maxval = np.amax(gray_frame)
    mean = np.mean(gray_frame)
    plt.figure()
    plt.imshow(gray_frame)
    
    
    template = np.zeros((37,37), np.uint8) + 1
    for i in range(len(template)):
        for j in range(len(template[0])):
            val = 0.1*(i-18)**2 + 0.5*(j-18)**2
            if val > 255:
                val = 255
            if len(gray_frame) < 200:
                template[i,j] *= int(val * mean / maxval) 
            elif len(gray_frame) >= 250:
                template[i,j] *= int(val)

    plt.figure()
    plt.imshow(template)

    res = cv2.matchTemplate(gray_frame, template, cv2.TM_SQDIFF_NORMED)
    plt.figure()
    plt.imshow(res)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    x_loc = min_loc[0] + 18
    y_loc = min_loc[1] + 18

    realtemplate = gray_frame[y_loc-25:y_loc+25,x_loc-25:x_loc+25]
    
    realtemplates.append(realtemplate)
    
    plt.figure()
    plt.imshow(realtemplate)
    plt.show()
