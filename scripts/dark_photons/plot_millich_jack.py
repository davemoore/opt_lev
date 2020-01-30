import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import scipy.stats as sp
#import bead_util as bu
from matplotlib.path import Path
import matplotlib.patches as patches
import _pickle as pickle

import matplotlib
matplotlib.rcParams['font.size'] = 15
#matplotlib.rc('font', family='serif') 
#matplotlib.rc('hatch', linewidth=2) 

name_list = ["lim_bead_comb_all_samesign.npy",
             "lim_bead_comb_all_opsign.npy",
             "lim_bead_comb_morp_samesign.npy",
             "lim_bead_comb_morp_opsign.npy"]

## output dictionary containing limits, for alex
output_dict = {}

# plt.figure()
# for f in name_list:

#     lims = np.load(f)
#     mid_val = len(lims)/2
#     plt.loglog( lims[:mid_val], lims[mid_val:] )



### comparison plot
fig=plt.figure()
ax = plt.gca()

dtype=(2,2)

ec = 'k'

# prev_col = [0.95, 0.95, 0.95]
# epsmin = 2.1e-7/5e3
# nmin = 0.1*1.0/7.9e17
# pp=ax.add_patch(Rectangle((epsmin, nmin), 1.0-2*epsmin, 2.5, ec='none', fc=prev_col, lw=2))
# plt.plot([epsmin,epsmin],[nmin, 1], 'k', dashes=[5,2], lw=2)
# ll=plt.plot([epsmin,1-epsmin],[nmin, nmin], 'k', dashes=[5,2], lw=2)
# xx = np.linspace(1e-11, epsmin, 1e2)
# yy = nmin * np.max(xx)/xx
# plt.loglog(xx,yy,'--', dashes=dtype, lw=2, color=ec)
# yy2 = nmin * (np.max(xx)/xx)**2
# plt.loglog(xx,yy2,'--', lw=2, color=ec)

prev_col = [0.85, 0.85, 0.85]

dtype=(1,0.5)
ax.add_patch(Rectangle((1.5e-5, 1.0/3.7e20), 1.0-1.5e-5, 2.5, ec=ec, fc=prev_col, lw=2))
epsmin = 1.5e-5
nmin = 1.0/3.7e20
xx = np.linspace(1e-11, epsmin, 1e2)
yy = nmin * np.max(xx)/xx
plt.loglog(xx,yy,'--', dashes=dtype, lw=2, color=ec)
yy2 = nmin * (np.max(xx)/xx)**2
plt.loglog(xx,yy2,'--', lw=2, color=ec)


# min_sens = 0.8e-5
# min_nuc = 1./8e15 * 0.1
# dtype=(1,0.5)
# ax.add_patch(Rectangle((min_sens, min_nuc), 1.0-min_sens, 2.5, ec=ec, fc=prev_col, lw=2))
# epsmin = min_sens
# nmin = min_nuc
# xx = np.linspace(1e-11, epsmin, 1e2)
# yy = nmin * np.max(xx)/xx
# #plt.loglog(xx,yy,'--', dashes=dtype, lw=2, color=ec)
# yy2 = nmin * (np.max(xx)/xx)**2
# #plt.loglog(xx,yy2,'--', lw=2, color=ec)

min_sens = 1e-5/16*3
min_nuc = 1./8e15*16 * 0.1
dtype=(1,0.5)
ax.add_patch(Rectangle((min_sens, min_nuc), 1.0-min_sens, 2.5, ec=ec, fc=prev_col, lw=2))
epsmin = min_sens
nmin = min_nuc
xx = np.linspace(1e-11, epsmin, 1e2)
yy = nmin * np.max(xx)/xx
plt.loglog(xx,yy,'--', dashes=dtype, lw=2, color=ec)
yy2 = nmin * (np.max(xx)/xx)**2
plt.loglog(xx,yy2,'--', lw=2, color=ec)


##### Us first ####
min_us  = 40e-6

prev_col = [0.5,0.75,1.0]
ec = [0,0,0.5]

## now solid
#good_vals = np.logical_and(lims[:mid_val] > min_us, lims[:mid_val] < 1)
lims = np.load("../paper_scripts/lim_bead_comb_all_samesign.npy")
mid_val = int(len(lims)/2)
lims = np.load("../paper_scripts/lim_bead_comb_all_opsign.npy")
mid_val = int(len(lims)/2)

verts = []
for x,y in zip(lims[:mid_val-1], lims[mid_val:-1]):
    if( x > min_us and x < 0.2 ):
        verts.append([x,y])
verts.append( [1-min_us, verts[-1][1]] )
verts.append( [1-min_us, 1.5] )
verts.append( [verts[0][0], 1.5] )
verts.append( [verts[0][0], verts[0][1]] )

path = Path(verts)
patch = patches.PathPatch(path, ec=ec, fc=prev_col, lw=1.5)
ax.add_patch(patch)

#output_dict = pickle.load(open("../paper_scripts/limits.pkl",'rb'))
with open("../paper_scripts/limits.pkl", 'rb') as f:
    output_dict = pickle.load(f, encoding='latin1')

## Perl
ax.add_patch(Rectangle((0.25, 1.9e-23), 0.5, 1.5, ec=ec, fc=prev_col, lw=1.5))

ep_vec = output_dict["perl_lims_op_sign"][0,:]
limit_vec = output_dict["perl_lims_op_sign"][1,:]

#plt.loglog( ep_vec, limit_vec, '--', dashes=dtype, lw=2, color=ec)
#plt.loglog( 1.-ep_vec, limit_vec, '--', dashes=dtype, lw=2, color=ec)

ep_vec = output_dict["perl_lims_same_sign"][0,:]
limit_vec_neut = output_dict["perl_lims_same_sign"][1,:]

#plt.loglog( ep_vec, limit_vec_neut, '--', lw=2, color=ec)
#plt.loglog( 1.-ep_vec, limit_vec_neut, '--', lw=2, color=ec)

##### Marinelli and Morpurgo ####
min_morp = 0.3
ax.add_patch(Rectangle((min_morp, 1.36e-21), 1.-2*min_morp, 1.5, ec=ec, fc=prev_col, lw=2))
lims = np.load("../paper_scripts/lim_bead_comb_morp_samesign.npy")
mid_val = int(len(lims)/2)
gpts = lims[:mid_val] < 0.5
plt.loglog( lims[:mid_val][gpts], lims[mid_val:][gpts], '--', dashes=dtype, lw=2, color=ec )
plt.loglog( 1.-lims[:mid_val][gpts], lims[mid_val:][gpts], '--', dashes=dtype, lw=2, color=ec )

epsmin = 1e-8
nmin = 1.85e-13 
xx = np.linspace(1e-11, epsmin, 1e2)
yy = nmin * np.max(xx)/xx
plt.loglog(xx,yy,'--', dashes=dtype, lw=2, color=ec)
nmin = 7.5e-7
yy2 = nmin * (np.max(xx)/xx)**2
plt.loglog(xx[:-19],yy2[:-19],'--', lw=2, color=ec)

output_dict["morp_lims_same_sign"] = np.vstack((lims[:mid_val][gpts], lims[mid_val:][gpts]))

lims = np.load("../paper_scripts/lim_bead_comb_morp_opsign.npy")
mid_val = int(len(lims)/2)

ep_vec = lims[:mid_val]*1.0
limit_vec_neut = lims[mid_val:]*1.0
gpts = np.logical_and(gpts, lims[mid_val:] < 0.9)  ## skip point with failed fit
plt.loglog( lims[:mid_val][gpts], lims[mid_val:][gpts], '--', lw=2, color=ec )
plt.loglog( 1.-lims[:mid_val][gpts], lims[mid_val:][gpts], '--', lw=2, color=ec )

output_dict["morp_lims_op_sign"] = np.vstack((lims[:mid_val][gpts], lims[mid_val:][gpts]))


#### Us again ###
#### Plot our lines last to make sure it's on top ####
lims = np.load("../paper_scripts/lim_bead_comb_all_samesign.npy")
mid_val = int(len(lims)/2)
#plt.loglog( lims[:mid_val-10], lims[mid_val:-10], '--', dashes=dtype, lw=2, color=ec )

output_dict["our_lims_same_sign"] = np.vstack((lims[:mid_val-10], lims[mid_val:-10]))

lims = np.load("../paper_scripts/lim_bead_comb_all_opsign.npy")
mid_val = int(len(lims)/2)
dtype2=[6,5]
#plt.loglog( lims[:mid_val-10], lims[mid_val:-10], '--', dashes=dtype2, lw=2, color=ec )

output_dict["our_lims_op_sign"] = np.vstack((lims[:mid_val-10], lims[mid_val:-10]))

plt.text(5.1e-5, 0.3, "Moore et al. (2014)", color=ec, fontsize=10, rotation=90, va='top', ha='left')
plt.text(0.125, 0.3, "Kim et al. (2007)", color=ec, fontsize=10, rotation=90, va='top', ha='left')
plt.text(0.35, 0.3, "Marinelli and Morpurgo (1982)", color=ec, fontsize=10, rotation=90, va='top', ha='left')
#plt.text(1.8e-5, 5e-21, "LHe (2 mm)", color='k', fontsize=12, rotation=0, va='bottom', ha='left')
#plt.text(0.8e-5, 1e-16, 1.5e-18/10, "SiO$_2$ (20 um)", color='k', fontsize=12, rotation=0, va='bottom', ha='left')
#plt.text(0.8e-5/16*3, 1e-16*16, 1.5e-18/10, "SiO$_2$ (5 um)", color='k', fontsize=12, rotation=0, va='bottom', ha='left')

#ll[0].set_zorder(0)
#pp.set_zorder(0)

ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_yticks([1, 1e-4, 1e-8, 1e-12, 1e-16, 1e-20, 1e-24])
plt.xlim([5e-7,1])
plt.ylim([1e-24, 1])
plt.xlabel("Fractional charge, $\epsilon$")
plt.ylabel("Abundance per nucleon, $n_\chi$")

for label in plt.gca().get_xticklabels()[1::2]:
    label.set_visible(False)
for label in plt.gca().get_yticklabels()[::2]:
    label.set_visible(False)

fig.set_size_inches(6,4.5)
plt.subplots_adjust(top=0.95, right=0.97, bottom=0.14, left=0.16)
plt.savefig("millich_proj_jack.pdf")


plt.show()

