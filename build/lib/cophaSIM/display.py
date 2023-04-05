# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:11:39 2023

@author: cpannetier
"""
from .config import NA,NT,NIN,NC,OW,InterfArray
from cophasim import config
import matplotlib.pyplot as plt
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

from cophasim.tol_colors import tol_cset
colors = tol_cset('muted')

SS = 12     # Small size
MS = 14     # Medium size
BS = 16     # Big size
figsize = (16,8)
rcParamsForBaselines = {"font.size":SS,
                        "axes.titlesize":SS,
                        "axes.labelsize":MS,
                        "axes.grid":True,
                        "xtick.labelsize":SS,
                        "ytick.labelsize":SS,
                        "legend.fontsize":SS,
                        "figure.titlesize":BS,
                        "figure.constrained_layout.use": False,
                        "figure.figsize":figsize,
                        'figure.subplot.hspace': 0.05,
                        'figure.subplot.wspace': 0,
                        'figure.subplot.left':0.1,
                        'figure.subplot.right':0.95
                        }


NINmes = config.FS['NINmes']

from matplotlib.ticker import AutoMinorLocator

# Each figure only shows 15 baselines, distributed on two subplots
# If there are more than 15 baselines, multiple figures will be created
NINdisp = 15
NumberOfBaseFiguresNIN = 1+NIN//NINdisp - 1*(NIN % NINdisp==0)
NumberOfBaseFigures = 1+NINmes//NINdisp - 1*(NINmes % NINdisp==0)

NAdisp = 10
NumberOfTelFigures = 1+NA//NAdisp - 1*(NA % NAdisp==0)
telcolors = colors[:NAdisp]*NumberOfTelFigures

def perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,RMSgdobs,RMSpdobs,
              generaltitle,savedir='',ext='pdf',infos={"details":''}):
        
    from simu import wl
    
    if (len(savedir)) and (not os.path.exists(savedir)):
        os.makedirs(savedir)
            
    plt.rcParams.update(rcParamsForBaselines)
    generaltitle = "GD and PD estimated"
    typeobs = "GDPDest"
    
    RMSgdobs = np.std(GDobs,axis=0)
    RMSpdobs = np.std(PDobs,axis=0)
    
    linestyles=[]
    if 'ThresholdGD' in config.FT.keys():
        linestyles.append(mlines.Line2D([],[], color='black',
                                    linestyle='--', label='Threshold GD'))
    
    for iFig in range(NumberOfBaseFigures):
        NINtodisplay=NINdisp
        if iFig == NumberOfBaseFigures-1:
            if (NINmes%NINdisp < NINdisp) and (NINmes%NINdisp != 0):
                NINtodisplay = NINmes%NINdisp
                
        iFirstBase = NINdisp*iFig   # Index of first baseline to display
        iLastBase = iFirstBase + NINtodisplay - 1        # Index of last baseline to display
        
        len2 = NINtodisplay//2 ; len1 = NINtodisplay-len2
        basecolors = colors[:len1]+colors[:len2]
        basecolors = np.array(basecolors)
        
        rangeBases = f"{baselines[iFirstBase]}-{baselines[iLastBase]}"
        title=f'{generaltitle}: {rangeBases}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=6,ncols=2, gridspec_kw={"height_ratios":[1,4,4,0.5,1,1]})
        ax1.set_title(f"From {baselines[iFirstBase]} \
to {baselines[iFirstBase+len1-1]}")
        ax6.set_title(f"From {baselines[iFirstBase+len1]} \
to {baselines[iLastBase]}")
        
        FirstSet = range(iFirstBase,iFirstBase+len1)
        SecondSet = range(iFirstBase+len1,iLastBase+1)
        iColor=0
        for iBase in FirstSet:   # First serie
            ax1.plot(timestamps,SNR[:,iBase],color=basecolors[iColor])
            if 'ThresholdGD' in config.FT.keys():
                ax1.hlines(config.FT['ThresholdGD'][iBase], timestamps[0],timestamps[-1], color=basecolors[iColor], linestyle='dashed')
            ax2.plot(timestamps,GDobs[:,iBase],color=basecolors[iColor])
            ax2.plot(timestamps,GDrefmic[:,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
            ax3.plot(timestamps,PDobs[:,iBase],color=basecolors[iColor])
            ax3.plot(timestamps,PDrefmic[:,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
            iColor+=1
        for iBase in SecondSet:   # Second serie
            ax6.plot(timestamps,SNR[:,iBase],color=basecolors[iColor])
            if 'ThresholdGD' in config.FT.keys():
                ax6.hlines(config.FT['ThresholdGD'][iBase],timestamps[0],timestamps[-1],color=basecolors[iColor], linestyle='dashed')
            ax7.plot(timestamps,GDobs[:,iBase],color=basecolors[iColor])
            ax7.plot(timestamps,GDrefmic[:,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
            ax8.plot(timestamps,PDobs[:,iBase],color=basecolors[iColor])
            ax8.plot(timestamps,PDrefmic[:,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
            iColor+=1
        
        ax4.bar(baselines[FirstSet],RMSgdobs[FirstSet], color=basecolors[:len1])
        # ax4.bar(baselines[FirstSet],simu.LR4[FirstSet],fill=False,edgecolor='black',linestyle='-')
        ax5.bar(baselines[FirstSet],RMSpdobs[FirstSet], color=basecolors[:len1])
        # ax5.bar(baselines[FirstSet],RMStrueOPD[FirstSet],fill=False,edgecolor='black',linestyle='-')
        
        ax9.bar(baselines[SecondSet],RMSgdobs[SecondSet], color=basecolors[len1:])
        # ax9.bar(baselines[SecondSet],simu.LR4[SecondSet],fill=False,edgecolor='black',linestyle='-')
        ax10.bar(baselines[SecondSet],RMSpdobs[SecondSet], color=basecolors[len1:])
        # ax10.bar(baselines[SecondSet],RMStrueOPD[SecondSet],fill=False,edgecolor='black',linestyle='-')
        
        ax1.get_shared_x_axes().join(ax1,ax2,ax3)
        ax6.get_shared_x_axes().join(ax6,ax7,ax8)
        ax4.get_shared_x_axes().join(ax4,ax5)
        ax9.get_shared_x_axes().join(ax9,ax10)
        
        ax1.get_shared_y_axes().join(ax1,ax6)
        ax2.get_shared_y_axes().join(ax2,ax7)
        ax3.get_shared_y_axes().join(ax3,ax8)
        ax4.get_shared_y_axes().join(ax4,ax9)
        ax5.get_shared_y_axes().join(ax5,ax10)
        
        ax6.tick_params(labelleft=False) ; ct.setaxelim(ax1,ydata=SNR,ymin=0)
        ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=GDobs[stationaryregim],ylim_min=[-wl/2,wl/2])
        ax8.tick_params(labelleft=False) ; ax3.set_ylim([-wl/2,wl/2])
        ax9.tick_params(labelleft=False) ; ct.setaxelim(ax4,ydata=np.concatenate([np.stack(RMSgdobs),[1]]),ymin=0)
        ax10.tick_params(labelleft=False) ; ct.setaxelim(ax5,ydata=np.concatenate([np.stack(RMSpdobs)]),ymin=0)
        
        ax4.tick_params(labelbottom=False)
        ax9.tick_params(labelbottom=False)
        
        ax1.set_ylabel('SNR')
        ax2.set_ylabel('Group-Delays [µm]')
        ax3.set_ylabel('Phase-Delays [µm]')
        ax4.set_ylabel('GD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
        ax5.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
        
        ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels
        
        ax3.set_xlabel('Time [s]', labelpad=-10) ; ax8.set_xlabel('Time [s]', labelpad=-10)
        ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')

        ax7.legend(handles=linestyles, loc='upper right')
        if display:
            fig.show()

        if len(savedir):
            if verbose:
                print("Saving perftable figure.")
            plt.savefig(savedir+f"Simulation{TimeID}_{typeobs}_{rangeBases}.{ext}")

    plt.rcParams.update(plt.rcParamsDefault)
