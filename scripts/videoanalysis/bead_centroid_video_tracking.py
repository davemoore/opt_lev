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
vfile1 = '/data/20151030/bead1/next_day/non_retarded/1v_ac_bias/1v_ac_bias.avi'
vfile2 = '/data/20151030/bead1/next_day/non_retarded/0_1v_ac_bias/0_1v_ac_bias.avi'
vfile3 = '/data/20151030/bead1/next_day/non_retarded/cant_sweep/cant_sweep.avi'
vfile4 = '/data/20151030/bead1/next_day/non_retarded/un_biased_no_drive/unbiased_no_drive.avi'
vfile5 = '/data/20151030/bead1/next_day/non_retarded/cant_sweep3/cant_sweep3.avi'
vfile6 = '/data/20151030/bead1/next_day/non_retarded/cant_sweep_long/cant_sweep_long.avi'

## File name, length of .h5 data files in seconds and number of 
## associated .h5 files (ie. numflashes)
files.append([vfile6, 50, 20, 'Cantilever Sweep'])
#files.append([vfile4, 50, 1, 'No Sweep, No Bias'])

nvideos = 0
videodicts = {}
for fil in files:
    new_dict = {}
    new_dict['handle'] = cv2.VideoCapture(fil[0])
    new_dict['fps'] = new_dict['handle'].get(5)
    new_dict['totframes'] = new_dict['handle'].get(7)
    new_dict['h5length'] = fil[1]
    new_dict['h5files'] = fil[2]
    new_dict['label'] = fil[3]
    videodicts[nvideos] = new_dict
    nvideos += 1

## Property IDs, video.get(num), obtained from docs.opencv.org/2.4/
##      modules/highgui/doc/reading_and_writing_images_and_video.html

##
## Helper functions for sub-pixel tracking
##

def polyfit2d(x, y, z, order=2):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m


def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

##
## Functions for use in video analysis
##

def load_video(filname, numflashes, h5length, label):
    videodict = {}
    videodict['handle'] = cv2.VideoCapture(filname)
    videodict['fps'] = new_dict['handle'].get(5)
    videodict['totframes'] = new_dict['handle'].get(7)
    videodict['h5length'] = h5length
    videodict['numflashes'] = numflashes
    videodict['label'] = label
    return videodict


def find_bead(video, tempsize=37):
    ## Making the bead_template by first finding the bead
    ## with a toy oval template

    ret, firstframe = video.read()

    gray_frame = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
    maxval = np.amax(gray_frame)
    mean = np.mean(gray_frame)
    #plt.figure()
    #plt.imshow(gray_frame)

    template = np.zeros((tempsize,tempsize), np.uint8) + 1
    for i in range(len(template)):
        for j in range(len(template[0])):
            val = 0.1*(i-(tempsize-1)*0.5)**2 + 0.5*(j-(tempsize-1)*0.5)**2
            if val > 255:
                val = 255
            if len(gray_frame) < 200:
                template[i,j] *= int(val * mean / maxval) 
            elif len(gray_frame) >= 250:
                template[i,j] *= int(val)

    res = cv2.matchTemplate(gray_frame, template, cv2.TM_SQDIFF_NORMED)
    #plt.figure()
    #plt.imshow(res)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    x_loc = min_loc[0] + (tempsize - 1)*0.5
    y_loc = min_loc[1] + (tempsize - 1)*0.5

    return x_loc, y_loc



def track_centroid(video, x_loc, y_loc, width=15):
    '''Track the centroid of the bead pattern in a video where the center
    of the bead is located at (x_loc, y_loc) and we consider a square image
    that is 2*width pixels on a side.'''
    xx = []
    yy = []
    totframes = video.get(7)
    while(1):
        ret, frame = video.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cut_frame = gray_frame[y_loc-width:y_loc+width,x_loc-width:x_loc+width]

        cut_frame = 255 - cut_frame
        M = cv2.moments(cut_frame)
        centroid_x = M['m10'] / M['m00']
        centroid_y = M['m01'] / M['m00']
        xx.append(centroid_x)
        yy.append(centroid_y)

        intensity.append(np.sum(gray_frame))
        k += 1

        if int((k/totframes)*100) >= l:
            print l
            l += 10

    xx = np.array(xx)
    yy = np.array(yy)

    return intensity, xx, yy



def proc_video(videodict, imagwidth=15, method=centroid):
    video = videodict['handle']
    fps = videodict['fps']
    totframes = videodict['totframes']
    h5length = videodict['h5length']
    numflashes = videodict['numflashes']
    label = videodict['label']

    x_loc, y_loc = find_bead(video)
    
    realtemplate = gray_frame[y_loc-imagwidth:y_loc+imagwidth,\
                              x_loc-imagwidth:x_loc+imagwidth]
    plt.figure()
    plt.imshow(realtemplate)
    plt.show()
    
    ##
    ## Now we loop over the image and track the bead
    ##

    intensity = []
    xx = []
    yy = []

    k = 0
    l = 0

    if method==centroid:
        intensity, xx, yy = track_centroid(video, x_loc, y_loc)
    elif method==template:

    ## Release the video file from active memory
    video.release()

    ## Post video-processing analysis such as cutting out the data with no drive
    ## and computing FFTs etc
    starts = []
    for i in range(numflashes):
        frames = int(fps * h5length)
        start = np.argmax(intensity[i*frames:(i+1)*frames])
        start += i*frames
        starts.append(start)

    ## Initialize arrays to be constructed
    xxf = np.array([])
    yyf = np.array([])
    xpsd = np.zeros((fps*50 - 1) * 0.5 + 1)
    ypsd = np.zeros((fps*50 - 1) * 0.5 + 1)

    ## Add to traces and PSDs for each .h5 file recorded by the DAQ
    for start in starts:
        s = int(start) + 1
        e = int(start + (fps*50))
        xxcurr = xx[s:e]
        yycurr = yy[s:e]
        xpsdcurr, freqs = mlab.psd(xxcurr, NFFT=len(xxcurr), Fs = fps)
        ypsdcurr, freqs = mlab.psd(yycurr, NFFT=len(yycurr), Fs = fps)
        xpsd = xpsd + xpsdcurr.T[0]
        ypsd = ypsd + ypsdcurr.T[0]
        #print xpsd.shape
        xxf = np.hstack((xxf, xxcurr))
        yyf = np.hstack((yyf, yycurr))
    xpsd *= (1. / len(starts))
    ypsd *= (1. / len(starts))

    t = np.linspace(0, len(xxf) * fps, len(xxf))

    output = {}
    output['ttrace'] = t
    output['xtrace'] = xxf
    output['ytrace'] = yyf
    output['freqs'] = freqs
    output['xpsd'] = xpsd
    output['ypsd'] = ypsd
    output['intensity'] = intensity

    return output






results = {}

for idx in range(nvideos):
    print files[idx][0]

    video = videodicts[idx]['handle']
    fps = videodicts[idx]['fps']
    totframes = videodicts[idx]['totframes']
    file_length = videodicts[idx]['h5length']
    num_files = videodicts[idx]['h5files']








for i in range(nvideos):
    
    t = results[i]['ttrace']
    xxf = results[i]['xtrace']
    yyf = results[i]['ytrace']
    freqs = results[i]['freqs']
    xpsd = results[i]['xpsd']
    ypsd = results[i]['ypsd']
    intensity = results[i]['intensity']

    intensity = np.array(intensity, dtype=float)

    xgrad = np.gradient(xxf)
    ygrad = np.gradient(yyf)
    intgrad = np.gradient(intensity)

    #plt.figure()
    #plt.plot(t, xxf - np.mean(xxf), label='x')
    #plt.plot(t, yyf - np.mean(yyf), label='y')
    title = videodicts[i]['label']
    #plt.title(title)
    #plt.legend(loc=0, numpoints=1)

    #plt.figure()
    #plt.plot(intensity / np.amax(intensity), label='intensity')
    #plt.plot(intgrad / np.amax(intgrad), label='intgrad')
    #plt.legend(loc=0, numpoints=1)

    plt.figure()
    plt.loglog(freqs, np.sqrt(xpsd), label='x')
    plt.loglog(freqs, np.sqrt(ypsd), label='y')
    plt.title(title)
    plt.legend(loc=0, numpoints=1)

plt.show()

    

    

