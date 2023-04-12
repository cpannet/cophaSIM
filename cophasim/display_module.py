# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:11:39 2023

@author: cpannetier
"""
from cophasim import config
from cophasim import coh_tools as ct
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
# import matplotlib.patches as mpatches
import numpy as np

from cophasim.tol_colors import tol_cset
colors = tol_cset('muted')

SS = 12     # Small size
MS = 14     # Medium size
BS = 16     # Big size
figsize = (16,8)
rcParamsForBaselines_withSNR = {"font.size":SS,
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

rcParamsForBaselines = {"font.size":SS,
                                   "axes.titlesize":SS,
                                   "axes.labelsize":MS,
                                   "axes.grid":True,
                                   "xtick.labelsize":SS,
                                   "ytick.labelsize":SS,
                                   "legend.fontsize":SS,
                                   "figure.titlesize":BS,
                                   "figure.constrained_layout.use": False,
                                   #"figure.constrained_layout.h_pad": 0.08,
                                   "figure.figsize":figsize,
                                   'figure.subplot.hspace': 0.05,
                                   'figure.subplot.wspace': 0,
                                   'figure.subplot.left':0.15,
                                   'figure.subplot.right':0.95
                                   }

rcParamsForSlides = {"font.size":SS,
                    "axes.titlesize":SS,
                    "axes.labelsize":MS,
                    "xtick.labelsize":SS,
                    "ytick.labelsize":SS,
                    "legend.fontsize":SS,
                    "figure.titlesize":BS,
                    "figure.constrained_layout.use": True,
                    "figure.figsize":figsize
                    }


global telescopes,baselines,closures,stationaryregim

from .config import NA,NIN
NINmes = config.FS['NINmes']

NINdisp = 15
NbOfBaseFiguresNIN = 1+NIN//NINdisp - 1*(NIN % NINdisp==0)
NbOfBaseFigures = 1+NINmes//NINdisp - 1*(NINmes % NINdisp==0)

NAdisp = 10
NbOfTelFigures = 1+NA//NAdisp - 1*(NA % NAdisp==0)
telcolors = colors[:NAdisp]*NbOfTelFigures


def perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,RMSgdobs,RMSpdobs,
              generalTitle,SNR=[],display=True,
              filename='',ext='pdf',infos={"details":''},verbose=False):
    
    global telescopes, baselines, closures, stationaryregim,wl,\
        PlotTel,PlotTelOrigin,PlotBaselineNIN,PlotBaseline,PlotClosure#,TelNameLength
            
    plt.rcParams.update(rcParamsForBaselines)

    plotSNR=False
    if len(SNR):
        plotSNR=True
        plt.rcParams.update(rcParamsForBaselines_withSNR)    

    linestyles=[]
    if 'ThresholdGD' in config.FT.keys():
        linestyles.append(mlines.Line2D([],[], color='black',
                                    linestyle='--', label='Threshold GD'))
    
    # Each figure only shows 15 baselines, distributed on two subplots
    # If there are more than 15 baselines, multiple figures will be created
    for iFig in range(NbOfBaseFigures):
        NINtodisplay=NINdisp
        if iFig == NbOfBaseFigures-1:
            if (NINmes%NINdisp < NINdisp) and (NINmes%NINdisp != 0):
                NINtodisplay = NINmes%NINdisp
                
        iFirstBase = NINdisp*iFig                       # Index of first baseline to display
        iLastBase = iFirstBase + NINtodisplay - 1       # Index of last baseline to display
        
        len2 = NINtodisplay//2 ; len1 = NINtodisplay-len2
        basecolors = colors[:len1]+colors[:len2]
        basecolors = np.array(basecolors)
        
        rangeBases = f"{baselines[iFirstBase]}-{baselines[iLastBase]}"
        title=f'{generalTitle}: {rangeBases}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        if plotSNR:
            (ax1,ax6),(ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=6,ncols=2, gridspec_kw={"height_ratios":[1,4,4,0.7,1,1]})
            ax1.set_title(f"From {baselines[iFirstBase]} \
to {baselines[iFirstBase+len1-1]}")
            ax6.set_title(f"From {baselines[iFirstBase+len1]} \
to {baselines[iLastBase]}")
        
        else:
            (ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=5,ncols=2, gridspec_kw={"height_ratios":[4,4,0.7,1,1]})
            ax2.set_title(f"From {baselines[iFirstBase]} \
to {baselines[iFirstBase+len1-1]}")
            ax7.set_title(f"From {baselines[iFirstBase+len1]} \
to {baselines[iLastBase]}")
        
        FirstSet = range(iFirstBase,iFirstBase+len1)
        SecondSet = range(iFirstBase+len1,iLastBase+1)
        iColor=0
        for iBase in FirstSet:   # First serie
            if plotSNR:
                ax1.plot(timestamps,SNR[:,iBase],color=basecolors[iColor])
            if 'ThresholdGD' in config.FT.keys():
                ax1.hlines(config.FT['ThresholdGD'][iBase], timestamps[0],timestamps[-1], color=basecolors[iColor], linestyle='dashed')
            ax2.plot(timestamps,GDobs[:,iBase],color=basecolors[iColor])
            ax2.plot(timestamps,GDrefmic[:,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
            ax3.plot(timestamps,PDobs[:,iBase],color=basecolors[iColor])
            ax3.plot(timestamps,PDrefmic[:,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
            iColor+=1
        for iBase in SecondSet:   # Second serie
            if plotSNR:
                ax6.plot(timestamps,SNR[:,iBase],color=basecolors[iColor])
            if 'ThresholdGD' in config.FT.keys():
                ax6.hlines(config.FT['ThresholdGD'][iBase],timestamps[0],timestamps[-1],color=basecolors[iColor], linestyle='dashed')
            ax7.plot(timestamps,GDobs[:,iBase],color=basecolors[iColor])
            ax7.plot(timestamps,GDrefmic[:,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
            ax8.plot(timestamps,PDobs[:,iBase],color=basecolors[iColor])
            ax8.plot(timestamps,PDrefmic[:,iBase],color=basecolors[iColor],linewidth=1, linestyle=':')
            iColor+=1
        
        ax4.bar(baselines[FirstSet],RMSgdobs[FirstSet], color=basecolors[:len1])
        ax5.bar(baselines[FirstSet],RMSpdobs[FirstSet], color=basecolors[:len1])
        
        ax9.bar(baselines[SecondSet],RMSgdobs[SecondSet], color=basecolors[len1:])
        ax10.bar(baselines[SecondSet],RMSpdobs[SecondSet], color=basecolors[len1:])
        
        if plotSNR:
            ax1.get_shared_x_axes().join(ax1,ax2,ax3)
            ax6.get_shared_x_axes().join(ax6,ax7,ax8)
        else:
            ax2.sharex(ax3)
            ax7.sharex(ax8)
            
        ax4.get_shared_x_axes().join(ax4,ax5)
        ax9.get_shared_x_axes().join(ax9,ax10)
        
        if plotSNR:
            ax1.get_shared_y_axes().join(ax1,ax6)
            ax6.tick_params(labelleft=False) ;
            ax1.set_ylabel('SNR')
            
        ax2.get_shared_y_axes().join(ax2,ax7)
        ax3.get_shared_y_axes().join(ax3,ax8)
        ax4.get_shared_y_axes().join(ax4,ax9)
        ax5.get_shared_y_axes().join(ax5,ax10)
        
        ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=GDobs[stationaryregim],ylim_min=[-wl/2,wl/2])
        ax8.tick_params(labelleft=False) ; ax3.set_ylim([-wl/2,wl/2])
        ax9.tick_params(labelleft=False) ; ct.setaxelim(ax4,ydata=np.concatenate([np.stack(RMSgdobs),[1]]),ymin=0)
        ax10.tick_params(labelleft=False) ; ct.setaxelim(ax5,ydata=np.concatenate([np.stack(RMSpdobs)]),ymin=0)
        
        ax4.tick_params(labelbottom=False)
        ax9.tick_params(labelbottom=False)
        
        
        ax2.set_ylabel('Group-Delays [µm]')
        ax3.set_ylabel('Phase-Delays [µm]')
        ax4.set_ylabel('GD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
        ax5.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60,loc='bottom')
        
        ax11.remove() ; ax12.remove()       # These axes are here to let space for ax3 and ax8 labels
        
        ax3.set_xlabel('Time [s]', labelpad=0) ; ax8.set_xlabel('Time [s]', labelpad=0)
        ax5.set_xlabel('Baselines') ; ax10.set_xlabel('Baselines')

        
        if display:
            fig.show()

        if len(filename):
            if verbose:
                print("Saving perftable figure.")
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(filename+f"_{rangeBases}.{extension}")
            else:
                plt.savefig(filename+f"_{rangeBases}.{ext}")

    plt.rcParams.update(plt.rcParamsDefault)



def simpleplot(timestamps, obs,rmsObs,generalTitle,plotObs,
               NameObs='PD',display=True,filename='',ext='pdf',infos={"details":''},
               verbose=False):
    
    global telescopes, baselines, closures, stationaryregim,wl,NINtodisplay#,TelNameLength,\
        #PlotTel,PlotTelOrigin,PlotBaselineNIN,PlotBaseline,PlotClosure
            # PlotBaselineIndex,PlotBaselineNINIndex
        
    plt.rcParams.update(rcParamsForBaselines)
    
    # Each figure only shows 15 baselines, distributed on two subplots
    # If there are more than 15 baselines, multiple figures will be created 
    # If there are less than 6 baselines, only one axe is plotted.
    # In between, two axes on a unique figure are plotted.

    NbOfObsToPlot = np.sum(plotObs)

    # NbOfFigures = 1 + NINtodisplay//NINdisp - 1*(NINtodisplay % NINdisp==0)
            
    plotObsIndex = np.argwhere(plotObs).ravel()
    for iFig in range(NbOfBaseFigures):

        NINtodisplay=NINdisp
        if iFig == NbOfBaseFigures-1:
            if (NINmes%NINdisp < NINdisp) and (NINmes%NINdisp != 0):
                NINtodisplay = NINmes%NINdisp
                
        iFirstBase = NINdisp*iFig                       # Index of first baseline to display
        iLastBase = iFirstBase + NINtodisplay - 1       # Index of last baseline to display
        
        len2 = NINtodisplay//2 ; len1 = NINtodisplay-len2
        basecolors = colors[:len1]+colors[:len2]
        basecolors = np.array(basecolors)
    
        rangeBases = f"{baselines[iFirstBase]}-{baselines[iLastBase]}"
        title=f'{generalTitle}: {rangeBases}'
        plt.close(title)
        fig=plt.figure(title, clear=True)
    
        if len(infos["details"]):
            fig.suptitle(title)

        OneAxe = False
        if NbOfObsToPlot <= 6:
            OneAxe=True

        if OneAxe:
            ax1,ax2 = fig.subplots(nrows=2,gridspec_kw={"height_ratios":[3,1]})
            basecolorstemp = basecolors[:NbOfObsToPlot]
            baselinestemp = [baselines[iBase] for iBase in plotObsIndex]
            
            iColor=0
            for iBase in plotObsIndex:
                ax1.plot(timestamps,obs[:,iBase],color=basecolorstemp[iColor],label=baselines[iBase])
                iColor+=1                
            
            p1=ax2.bar(baselinestemp,[rmsObs[iBase] for iBase in plotObsIndex], color=basecolorstemp)
            
            ax2.bar_label(p1,label_type='edge',fmt='%.2f')
            ax2.set_box_aspect(1/20)
            ax1.legend()
    
        else:
            (ax1,ax3),(ax2,ax4) = fig.subplots(nrows=2,ncols=2, sharey='row',gridspec_kw={"height_ratios":[3,1]})
            ax1.set_title(f"From {baselines[iFirstBase]} \
to {baselines[iFirstBase+len1-1]}")
            ax3.set_title(f"From {baselines[iFirstBase+len1]} \
to {baselines[iLastBase]}")
            
            FirstSet = range(iFirstBase,iFirstBase+len1)
            SecondSet = range(iFirstBase+len1,iLastBase+1)
            NbOfObs = len(FirstSet)+len(SecondSet)
            barbasecolors = ['grey']*NbOfObs
            
            iColor=0
            for iBase in FirstSet:   # First serie
                if PlotBaseline[iBase]:
                    ax1.plot(timestamps,obs[:,iBase],color=basecolors[iColor])
                    barbasecolors[iColor] = basecolors[iColor]
                iColor+=1
                
            for iBase in SecondSet:   # Second serie
                if PlotBaseline[iBase]:
                    ax3.plot(timestamps,obs[:,iBase],color=basecolors[iColor])
                    barbasecolors[iColor] = basecolors[iColor]
                iColor+=1
                
            p1=ax2.bar(baselines[FirstSet],rmsObs[FirstSet], color=barbasecolors[:len1])
            p2=ax4.bar(baselines[SecondSet],rmsObs[SecondSet], color=barbasecolors[len1:])
            ax4.sharey(ax2) ; ax4.tick_params(labelleft=False)
            ct.setaxelim(ax1, ydata=obs, ymargin=0.4,ymin=0)
            if NameObs=='PD':
                ax4.set_ylim([0,wl])
            else:
                ct.setaxelim(ax4, ydata=rmsObs,ymin=0)
                
            ax2.bar_label(p1,label_type='edge',fmt='%.2f')
            ax4.bar_label(p2,label_type='edge',fmt='%.2f')
            
            
        ct.setaxelim(ax1,ydata=[obs[:,iBase] for iBase in plotObsIndex])
        ct.setaxelim(ax2,ydata=list(rmsObs)+[wl/2], ymin=0)
        ax1.set_ylabel(f'{NameObs} [µm]')
        ax1.set_xlabel("Time [s]") ; ax3.set_xlabel("Time [s]")
        ax2.set_ylabel('RMS [µm]')
        ax2.set_xlabel("Baselines") ; ax4.set_xlabel("Baselines")
        ax2.set_anchor('S') ; ax4.set_anchor('S')
        ax2.set_box_aspect(1/6) ; ax4.set_box_aspect(1/6)
    
        if display:
            fig.show()
            
        if len(filename):
            if verbose:
                print("Saving perftable figure.")
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(filename+f"_{rangeBases}.{extension}")
            else:
                plt.savefig(filename+f"_{rangeBases}.{ext}")
                
        # figname = '_'.join(title.split(' ')[:3])
        # figname = f"SNRonly"
        # if display:
        #     if pause:
        #         plt.pause(0.1)
        #     else:
        #         plt.show()  
                
        # if len(savedir):
        #     fig.savefig(savedir+f"Simulation{TimeID}_opd.{ext}")
            
        #     if isinstance(ext,list):
        #         for extension in ext:
        #             plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
        #     else:
        #         plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
            
        #     if figsave:
        #         prefix = details.replace("=","").replace(";","").replace(" ","").replace(".","").replace('\n','_').replace('Phase-delay','PD').replace('Group-delay','GD')
        #         if isinstance(ext,list):
        #             for extension in ext:
        #                 plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
        #         else:
        #             plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
        #     fig.show()
            
            
            
    # OneAxe = False ; NbOfBaselinesToPlot = np.sum(PlotBaseline)
    # if NbOfBaselinesToPlot < 8:
    #     OneAxe=True
    #     ax1, ax2 = fig.subplots(nrows=2,ncols=1, gridspec_kw={"height_ratios":[3,1]})
    # else:
    #     len1 = NbOfBaselinesToPlot//2 ; len2 = NbOfBaselinesToPlot%2
    #     (ax1,ax3),(ax2,ax4) = fig.subplots(nrows=2,ncols=2, gridspec_kw={"height_ratios":[3,1]})
    #     ax1.set_title("First serie of baselines")
    #     ax3.set_title("Second serie of baselines")
    
    # setaxelim(ax1, ydata=1/averagePdVar[:,PlotBaseline], ymargin=0.2, ymin=0)
    # setaxelim(ax2, ydata=MeanSquaredSNR[PlotBaseline]+RMSSquaredSNR[PlotBaseline], ymargin=0.2,ymin=0)

    # ax1.set_xlabel("Time (s)")
    # ax1.set_ylabel('SNR²')
    # ax2.set_ylabel('<SNR²>')
    
    # if OneAxe:
    #     baselinestemp = [baselines[iBase] for iBase in PlotBaselineIndex]
    #     basecolorstemp = basecolors[:NbOfBaselinesToPlot]
    #     barbasecolors = ['grey']*NIN
        
    #     ax2.set_anchor('S')
    #     ax2.set_box_aspect(1/15)
        
    #     k=0
    #     for iBase in PlotBaselineIndex:   # First serie
    #         ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolorstemp[k])
    #         barbasecolors[iBase] = basecolorstemp[k]
    #         k+=1
            
    #     p1=ax2.bar(baselines,MeanSquaredSNR, color=barbasecolors, yerr=RMSSquaredSNR)
    
    # else:    
    #     for iBase in range(len1):   # First serie
    #         if PlotBaseline[iBase]:
    #             ax1.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])

    #     for iBase in range(len1,NIN):   # Second serie
    #         if PlotBaseline[iBase]:
    #             ax3.plot(t[timerange],1/averagePdVar[timerange,iBase],color=basecolors[iBase])
        
    #     p1=ax2.bar(baselines[:len1],MeanSquaredSNR[:len1], color=basecolors[:len1], yerr=RMSSquaredSNR[:len1])
    #     p2=ax4.bar(baselines[len1:],MeanSquaredSNR[len1:], color=basecolors[len1:], yerr=RMSSquaredSNR[len1])

    #     ax3.sharex(ax1)
    #     ax3.sharey(ax1) ; ax3.tick_params(labelleft=False)
    #     ax4.sharey(ax2) ; ax4.tick_params(labelleft=False)
    #     ax4.bar_label(p2,label_type='edge',fmt='%.2f')
    #     ax3.set_xlabel("Time (s)")#, labelpad=xlabelpad)
    #     ax2.set_anchor('S') ; ax4.set_anchor('S')
    #     ax2.set_box_aspect(1/8) ; ax4.set_box_aspect(1/8)

    # ax2.bar_label(p1,label_type='edge',fmt='%.2f')


    # figname = '_'.join(title.split(' ')[:3])
    # figname = f"SNRonly"
    # if figsave:
    #     if isinstance(ext,list):
    #         for extension in ext:
    #             plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
    #     else:
    #         plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
    # fig.show()



