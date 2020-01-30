import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

import matplotlib
matplotlib.rcParams['font.size'] = 15

class MyLogFormatter(matplotlib.ticker.LogFormatterMathtext):
    def __call__(self,x,pos=None):
        rv = '-'+matplotlib.ticker.LogFormatterMathtext.__call__(self,x,pos)
        return rv


def make_pos_plot(dq, N, Z, linecol='k', fillcol='k', do_fill = True, linesty="-", lw=1.5):


    if( linecol != "k"): lw = 2

    min_val, max_val = 1e-23, 5e-20 #1e-28, 1e-18
    min_exp, max_exp = -23, -17

    epvec = np.logspace(min_exp,max_exp,1e3)
    envec = np.logspace(min_exp,max_exp,1e3)

    alpha = 1.0
    plt.subplot(2,1,1)

    if(Z > 0):
        yvals = (dq-N*envec)/Z
        yvals[yvals<min_val] = 0.5*min_val
        if( linesty != ":"):
            plt.loglog( envec, yvals, color=linecol, linewidth=lw, linestyle=linesty)
        else:
            plt.loglog( envec, yvals, color=linecol, linewidth=lw, linestyle=linesty, dashes=[1,1])
        if(do_fill):
            plt.fill_between( envec, yvals, np.ones_like(envec)*max_val, color=fillcol, edgecolor='none', alpha=alpha, linestyle=linesty )
    else:
        plt.loglog( [dq, dq], [min_val, max_val], color=linecol, linewidth=lw, linestyle=linesty)
        plt.fill_between( [dq, max_val], [min_val, min_val], [max_val, max_val], color=fillcol, edgecolor='none', alpha=alpha )
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])

    plt.subplot(2,1,2)
    if(Z > 0):
        yvals = (N*envec-dq)/Z
        yvals[yvals < min_val] = 0.5*min_val
        if( linesty != ":"):
            plt.loglog( envec, yvals, color=linecol, linewidth=lw, linestyle=linesty)
        else:
            plt.loglog( envec, yvals, color=linecol, linewidth=lw, linestyle=linesty, dashes=[1,1])
        if(do_fill):
            plt.fill_between( envec, np.ones_like(envec)*min_val, yvals, color=fillcol, edgecolor='none', alpha=alpha )
        if(linesty != ":" ):
            plt.loglog( envec, (N*envec+dq)/Z, color=linecol, linewidth=lw, linestyle=linesty)
        else:
            plt.loglog( envec, (N*envec+dq)/Z, color=linecol, linewidth=lw, linestyle=linesty, dashes=[1,1])
        yvals = (N*envec+dq)/Z
        yvals[yvals < min_val] = 0.5*min_val
        if(do_fill):
            plt.fill_between( envec, yvals, np.ones_like(envec)*max_val, color=fillcol, edgecolor='none', alpha=alpha )
    else:
        plt.loglog( [dq, dq], [min_val, max_val], color=linecol, linewidth=lw)
        plt.fill_between( [dq, max_val], [min_val, min_val], [max_val, max_val], color=fillcol, edgecolor='none', alpha=alpha )

    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.gca().invert_yaxis()

#yel=[255./256,246./256,143/256.]
yel = [0.5,0.75,1.0]
ec = [0,0,0.5]

prev_col = [0.85, 0.85, 0.85]
light_col = [0.95, 0.95, 0.95]

our_col = [0.7,0.4,0.4]
our_ec=[0.5,0.,0]

our_col2 = [0.65,0.35,0.35]

prev_dat = [ #[2.6e-21, 2, 2],
[5.1e-23/2, 2.0, 2.0, True, prev_col,':','k'],  ###  Update these to more reasonable 
             [1.2e-20/2, 30.09, 30.0, True, our_col,'-',our_ec],  ###  Update these to more reasonable
             [1.1e-21/2, 30.09, 30.0, True, our_col2,':',our_ec],  ###  Update these to more reasonable 
             #[1.6e-28, 30.09, 30.0, True, light_col,':','k'],  ###  Update these to more reasonable 
             #[8.4e-25, 30.09, 30.0, True, prev_col,'-','k'],   ###  values based on our sensitivity
             [1.5e-21, 1, 0, True, yel,'-',ec],
             #[1.6e-21, 29.5, 25.8],
             [1.0e-21, 76.0, 70.0, True, yel,'-',ec],
            ]


fig=plt.figure()

for dat in prev_dat:

    N = dat[1]
    Z = dat[2]
    dq = dat[0]*(N+Z)

    make_pos_plot(dq, N, Z, do_fill=dat[3],fillcol=dat[4],linesty=dat[5],linecol=dat[6])

plt.subplot(2,1,1)
ax = plt.gca()
#ax.set_xticks([1e-27,1e-25, 1e-23, 1e-21, 1e-19])
ax.set_xticklabels([])
#ax.set_yticks([1e-27,1e-25, 1e-23, 1e-21, 1e-19])
#for label in plt.gca().get_yticklabels()[-2:]:
#    label.set_visible(False)

plt.text(2e-23, 2.5e-21, "Bressi et al. (2011)", color=ec, fontsize=12, rotation=0, va='bottom', ha='left')
plt.text(4e-20, 4e-20, "Baumann et al.\n(1988)", color=ec, fontsize=12, rotation=0, va='top', ha='right')
#plt.text(2e-24, 6e-23, "This proposal", color='k', fontsize=13, rotation=0, va='bottom', ha='left')
#plt.text(2e-29, 5e-28, "Ultimate", color='k', fontsize=13, rotation=0, va='bottom', ha='left')

plt.subplot(2,1,2)
ax = plt.gca()
#ax.set_xticks([1e-27,1e-25, 1e-23, 1e-21, 1e-19])
#ax.yaxis.set_ticklabels(['',1e-23,1e-22, 1e-21,''])
ax.yaxis.set_major_formatter(MyLogFormatter())
#for label in plt.gca().get_xticklabels()[1::4]:
#    label.set_visible(False)
for label in plt.gca().get_yticklabels()[1::5]:
     print(label)
     label.set_visible(False)

fig.text(0.005, 0.5, "Proton + electron charge, $q_p\ +\ q_e$ [$e$]", va='center', rotation='vertical')
plt.xlabel("Neutron charge, $q_n$ [$e$]")

fig.set_size_inches(6,4.5)
plt.subplots_adjust( left=0.18, bottom=0.13, right=0.95, top=0.95, hspace=0.025 )
plt.savefig("matter_neutrality_jack.pdf")

plt.show()
