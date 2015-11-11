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
vfile2 = '/data/20151030/bead1/cant_sweep_2_11Hz_camera_beta1e7/cant_sweep_2_11Hz_cant_sweep_beta1e7.avi'

## File name, length of .h5 data files in seconds and number of 
## associated .h5 files (ie. numflashes)
files.append([vfile1, 50, 5, 'no razer, 3.7 Hz'])
files.append([vfile2, 50, 5, 'razer, 11 Hz'])

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


results = {}

for idx in range(nvideos):
    print files[idx][0]

    video = videodicts[idx]['handle']
    fps = videodicts[idx]['fps']
    totframes = videodicts[idx]['totframes']
    file_length = videodicts[idx]['h5length']
    num_files = videodicts[idx]['h5files']

    ##
    ## Making the bead_template by first finding the bead
    ## with a toy circular template
    ##

    ret, firstframe = video.read()

    gray_frame = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
    maxval = np.amax(gray_frame)
    mean = np.mean(gray_frame)
    #plt.figure()
    #plt.imshow(gray_frame)
    
    template = np.zeros((31,31), np.uint8) + 1
    for i in range(len(template)):
        for j in range(len(template[0])):
            val = 0.1*(i-15)**2 + 0.5*(j-15)**2
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

    x_loc = min_loc[0] + 15
    y_loc = min_loc[1] + 15

    realtemplate = gray_frame[y_loc-25:y_loc+25,x_loc-25:x_loc+25]
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
    while(1):
        ret, frame = video.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_frame, realtemplate, cv2.TM_SQDIFF_NORMED)

        #plt.figure()
        #plt.imshow(res)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #print min_loc

        x_loc = min_loc[0] #+ len(realtemplate) / 2
        y_loc = min_loc[1] #+ len(realtemplate[0]) / 2

        subres = res[y_loc-20:y_loc+20,x_loc-20:x_loc+20]

        #plt.figure()
        #plt.imshow(subres)
        #plt.show()

        y = np.linspace(0,len(subres)-1,len(subres))
        x = np.linspace(0,len(subres[0])-1, len(subres[0]))

        #print len(x), len(y), len(subres), len(subres[0])

        #m = polyfit2d(x,y,subres)

        #funcvals = polyval2d(x, y, m)
        #print x, y, funcvals

        interpobj = interp.RectBivariateSpline(x,y,subres)
        def interpfunc(xi):
            x = xi[0]
            y = xi[1]
            return interpobj(x,y)[0][0]

        fit = opti.minimize(interpfunc, [15, 15])
        bead_loc = fit.x

        xx.append(bead_loc[0])
        yy.append(bead_loc[1])


        intensity.append(np.sum(gray_frame))
        k += 1

        if int((k/totframes)*100) >= l:
            print l
            l += 10

    starts = []
    for i in range(num_files):
        frames = int(fps * file_length)
        start = np.argmax(intensity[i*frames:(i+1)*frames])
        start += i*frames
        starts.append(start)
    print starts

    xxf = np.array([])
    yyf = np.array([])

    for start in starts:
        s = int(start)
        e = int(start + (fps*50))
        xxcurr = xx[s:e]
        yycurr = yy[s:e]
        xxf = np.hstack((xxf, xxcurr))
        yyf = np.hstack((yyf, yycurr))

    t = np.linspace(0, len(xxf) * fps, len(xxf))

    xpsd, freqs = mlab.psd(xxf, NFFT=len(xxf), Fs = fps)
    ypsd, freqs = mlab.psd(yyf, NFFT=len(yyf), Fs = fps)

    output = {}
    output['ttrace'] = t
    output['xtrace'] = xxf
    output['ytrace'] = yyf
    output['freqs'] = freqs
    output['xpsd'] = xpsd
    output['ypsd'] = ypsd
    
    results[idx] = output

    video.release()


for i in range(nvideos):
    
    t = results[i]['ttrace']
    xxf = results[i]['xtrace']
    yyf = results[i]['ytrace']
    freqs = results[i]['freqs']
    xpsd = results[i]['xpsd']
    ypsd = results[i]['ypsd']

    plt.figure()
    plt.plot(t, xxf - np.mean(xxf), label='x')
    plt.plot(t, yyf - np.mean(yyf), label='y')
    title = videodicts[i]['label']
    plt.title(title)
    plt.legend(loc=0, numpoints=1)

    plt.figure()
    plt.loglog(freqs, xpsd, label='x')
    plt.loglog(freqs, ypsd, label='y')
    plt.title(title)
    plt.legend(loc=0, numpoints=1)

plt.show()

    

    
