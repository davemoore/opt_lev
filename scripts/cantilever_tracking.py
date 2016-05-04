import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.misc as misc
import scipy.interpolate as interp

data_dir = "/data/20160329/bead1/recenter/cantilever_images"

positions = [20, 60, 100, 150]
realpositions = []

onlyfiles = [f for f in os.listdir(data_dir) if \
             os.path.isfile(os.path.join(data_dir, f))]

realdist = []
pixeldist = []

def add_loc(event):
    global pixeldist
    pixeldist.append(event.ydata)
    plt.clf()

def take_pic(pic,data_dir):
    curr_image = misc.imread(os.path.join(data_dir, pic))
    small_image = curr_image[:600,:800]
    fig = plt.figure(figsize=(5,5),dpi=150)
    ax = fig.add_subplot(111)
    ax.imshow(small_image)
    cid = fig.canvas.mpl_connect('button_press_event',add_loc)
    plt.show()    

cent_files = ['trap_cent_' + str(voltage) + 'V.png' \
              for voltage in [10,5,0]]
    
    
for pic in cent_files:
    voltage = re.findall('\d+V.png',pic)[0][:-5]
    dist = 80 - float(voltage) * 8.
    realdist.append(dist)
        
    take_pic(pic,data_dir)

#print realdist, pixeldist

interpfunc = interp.interp1d(pixeldist, realdist)
realdist = []
pixeldist = []

for pos in positions:
    pic0 = 'cant_' + str(pos) + 'um_pos_10V.png'
    pic40 = 'cant_' + str(pos) + 'um_pos_5V.png'
    pic80 = 'cant_' + str(pos) + 'um_pos_0V.png'

    take_pic(pic0,data_dir)
    realdist.append(interpfunc(pixeldist[0]))

    if pos == positions[-1]:
        realpositions.append(realdist[0])
        break

    take_pic(pic40,data_dir)
    realdist.append(realdist[0]+40)

    take_pic(pic80,data_dir)
    realdist.append(realdist[0]+80)

    realpositions.append(realdist[0])
 
    interpfunc = interp.interp1d(pixeldist, realdist)
    realdist = []
    pixeldist = []

positions = np.array(positions)
realpositions = np.array(realpositions)

print positions, realpositions

'''
for pic in onlyfiles:
    for pos in positions:
        if pos == 0:
            curr_pos = 'cent'
        else:
            curr_pos = str(pos) + 'um'
    curr_image = scipy.misc.imread(pic)
'''
