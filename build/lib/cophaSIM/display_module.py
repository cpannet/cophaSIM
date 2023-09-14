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
                        'figure.subplot.left':0.15,
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

rcParamsForOneAxe = {"font.size":SS,
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
                                   'figure.subplot.wspace': 0.05,
                                   'figure.subplot.left':0.05,
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


global telescopes,baselines,closures

from .config import NA,NIN,ND
NINmes = config.FS['NINmes']
NCmes = config.FS['NCmes']
NCmes = ND

NCdisp = 20
nClosureFiguresNC = 1+ND//NCdisp - 1*(ND % NCdisp==0)
nClosureFigures = 1+NCmes//NCdisp - 1*(NCmes % NCdisp==0)

NINdisp = 15
nBaseFiguresNIN = 1+NIN//NINdisp - 1*(NIN % NINdisp==0)
nBaseFigures = 1+NINmes//NINdisp - 1*(NINmes % NINdisp==0)

NAdisp = 10
nTelFigures = 1+NA//NAdisp - 1*(NA % NAdisp==0)
telcolors = colors[:NAdisp]*nTelFigures

wl = config.wlOfTrack

def perftable(timestamps, PDobs,GDobs,GDrefmic,PDrefmic,RMSgdobs,RMSpdobs,
              plotObs,generalTitle,SNR=[],obsType='',display=True,
              filename='',ext='pdf',infos={"details":''},verbose=False):
    
    global telescopes, baselines, closures,wl,\
        PlotTel,PlotTelOrigin,plotBaselineNIN,plotBaseline,PlotClosure#,TelNameLength
                
    nObs = NINmes
    nCurvesMaxPerFigure = NINdisp
    nFigures = nBaseFigures
    curvesNames = baselines
    
    plt.rcParams.update(rcParamsForOneAxe)

    nObsToPlot = np.sum(plotObs)
    plotObsIndex = np.argwhere(plotObs).ravel()

    plotSNR=False
    if len(SNR):
        plotSNR=True
        linestyles=[mlines.Line2D([],[], color='black',
                                  linestyle='--', label='Threshold GD')]

    
    # Each figure only shows 15 baselines, distributed on two subplots
    # If there are more than 15 baselines, multiple figures will be created
    for iFig in range(nFigures):
        nCurvesToDisplay=nCurvesMaxPerFigure
        if iFig == nFigures-1:
            nCurvesOnLastFigure = nObs%nCurvesMaxPerFigure
            if (nCurvesOnLastFigure < nCurvesMaxPerFigure) and (nCurvesOnLastFigure != 0):
                nCurvesToDisplay = nCurvesOnLastFigure
                
        iFirstCurve = nCurvesMaxPerFigure*iFig                       # Index of first baseline to display
        iLastBase = iFirstCurve + nCurvesToDisplay - 1       # Index of last baseline to display
        
        len2 = nCurvesToDisplay//2 ; len1 = nCurvesToDisplay-len2
        colorsArray = colors[:len1]+colors[:len2]
        colorsArray = np.array(colorsArray)
        
        oneAxe = False
        if nObsToPlot <= 6:
            oneAxe=True
    
        if not oneAxe:
            rangeCurves = f"{curvesNames[iFirstCurve]}-{curvesNames[iLastBase]}"
            title=f'{generalTitle}: {rangeCurves}'
        else:
            rangeCurves = ""
            title=generalTitle

        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        
        if plotSNR:
            axGhost=[0]*5       # will contain axes that I remove directly after creation
            
            if oneAxe:
                plt.rcParams.update(rcParamsForOneAxe)

                axs = fig.subplots(nrows=3,ncols=4, gridspec_kw={"height_ratios":[1,3,3],"width_ratios":[8,0.5,1,1]})
                
                ax1,ax2,ax3 = axs[:,0]
                
                gs = axs[0, 1].get_gridspec()
                # remove the underlying axes
                for ax in axs[0, 1:]:
                    ax.remove()
                axLegend = fig.add_subplot(gs[0, 1:])
                
                gs = axs[1, 2].get_gridspec()
                # remove the underlying axes
                for ax in axs[1:, 2]:
                    ax.remove()
                ax4 = fig.add_subplot(gs[1:, 2])
                
                gs = axs[1, 3].get_gridspec()
                # remove the underlying axes
                for ax in axs[1:, 3]:
                    ax.remove()
                ax5 = fig.add_subplot(gs[1:, 3])
                
                for ax in axs[1:, 1]:
                    ax.axis("off")
                
            else:
                plt.rcParams.update(rcParamsForBaselines_withSNR)
                (ax1,ax6,axLegend),(ax2,ax7,axGhost[0]), \
                    (ax3,ax8,axGhost[1]),(ax11,ax12,axGhost[2]),\
                        (ax4,ax9,axGhost[3]),(ax5,ax10,axGhost[4]) = fig.subplots(nrows=6,ncols=3, gridspec_kw={"height_ratios":[1,4,4,0.7,1,1],"width_ratios":[5,5,1]})
                
                ax1.set_title(f"From {curvesNames[iFirstCurve]} \
    to {curvesNames[iFirstCurve+len1-1]}")
                ax6.set_title(f"From {curvesNames[iFirstCurve+len1]} \
    to {curvesNames[iLastBase]}")
                        
                for ax in axGhost:  # removal of the axes
                    ax.remove()
            
            axLegend.axis("off")    # extinction of the axes of axLegend
    
        else:
            plt.rcParams.update(rcParamsForBaselines)
            (ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=5,ncols=2, gridspec_kw={"height_ratios":[4,4,0.7,1,1]})
            ax2.set_title(f"From {curvesNames[iFirstCurve]} \
to {curvesNames[iFirstCurve+len1-1]}")
            ax7.set_title(f"From {curvesNames[iFirstCurve+len1]} \
to {curvesNames[iLastBase]}")
        
        
        if oneAxe:
            colorsArrayOneAxe = colorsArray[:nObsToPlot]
            curvesNamesOneAxe = [curvesNames[iCurve] for iCurve in plotObsIndex]
            iColor=0
            for iCurve in plotObsIndex:   # First serie
                if plotSNR:
                    ax1.plot(timestamps,SNR[:,iCurve],color=colorsArray[iColor])
                    ax1.hlines(config.FT['ThresholdGD'][iCurve], timestamps[0],timestamps[-1], color=colorsArray[iColor], linestyle='dashed')
                ax2.plot(timestamps,GDobs[:,iCurve],color=colorsArray[iColor])
                ax2.plot(timestamps,GDrefmic[:,iCurve],color=colorsArray[iColor],linewidth=1, linestyle=':')
                ax3.plot(timestamps,PDobs[:,iCurve],color=colorsArray[iColor])
                ax3.plot(timestamps,PDrefmic[:,iCurve],color=colorsArray[iColor],linewidth=1, linestyle=':')
                iColor+=1

            p1=ax4.barh(curvesNamesOneAxe,[RMSgdobs[iCurve] for iCurve in plotObsIndex], color=colorsArrayOneAxe)
            p3=ax5.barh(curvesNamesOneAxe,[RMSpdobs[iCurve] for iCurve in plotObsIndex], color=colorsArrayOneAxe)
            
            
        else:
            FirstSet = range(iFirstCurve,iFirstCurve+len1)
            SecondSet = range(iFirstCurve+len1,iLastBase+1)
            iColor=0
            for iCurve in FirstSet:   # First serie
                if plotSNR:
                    ax1.plot(timestamps,SNR[:,iCurve],color=colorsArray[iColor])
                    ax1.hlines(config.FT['ThresholdGD'][iCurve], timestamps[0],timestamps[-1], color=colorsArray[iColor], linestyle='dashed')
                ax2.plot(timestamps,GDobs[:,iCurve],color=colorsArray[iColor])
                ax2.plot(timestamps,GDrefmic[:,iCurve],color=colorsArray[iColor],linewidth=1, linestyle=':')
                ax3.plot(timestamps,PDobs[:,iCurve],color=colorsArray[iColor])
                ax3.plot(timestamps,PDrefmic[:,iCurve],color=colorsArray[iColor],linewidth=1, linestyle=':')
                iColor+=1
            for iCurve in SecondSet:   # Second serie
                if plotSNR:
                    ax6.plot(timestamps,SNR[:,iCurve],color=colorsArray[iColor])
                    ax6.hlines(config.FT['ThresholdGD'][iCurve],timestamps[0],timestamps[-1],color=colorsArray[iColor], linestyle='dashed')
                ax7.plot(timestamps,GDobs[:,iCurve],color=colorsArray[iColor])
                ax7.plot(timestamps,GDrefmic[:,iCurve],color=colorsArray[iColor],linewidth=1, linestyle=':')
                ax8.plot(timestamps,PDobs[:,iCurve],color=colorsArray[iColor])
                ax8.plot(timestamps,PDrefmic[:,iCurve],color=colorsArray[iColor],linewidth=1, linestyle=':')
                iColor+=1
        
            p1=ax4.bar(curvesNames[FirstSet],RMSgdobs[FirstSet], color=colorsArray[:len1])
            p3=ax5.bar(curvesNames[FirstSet],RMSpdobs[FirstSet], color=colorsArray[:len1])
            
            p2=ax9.bar(curvesNames[SecondSet],RMSgdobs[SecondSet], color=colorsArray[len1:])
            p4=ax10.bar(curvesNames[SecondSet],RMSpdobs[SecondSet], color=colorsArray[len1:])
        
        """
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
            axLegend.legend(handles = linestyles)
            
        ax2.get_shared_y_axes().join(ax2,ax7)
        ax3.get_shared_y_axes().join(ax3,ax8)
        ax4.get_shared_y_axes().join(ax4,ax9)
        ax5.get_shared_y_axes().join(ax5,ax10)
        
        ax7.tick_params(labelleft=False) ; ct.setaxelim(ax2,ydata=GDobs,ylim_min=[-wl/2,wl/2])
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
        """
        
        
        """
        Tune the axis
        """
               
        if plotSNR:
            ax1.get_shared_x_axes().join(ax1,ax2,ax3)
            ax1.set_ylabel('SNR')
            axLegend.legend(handles = linestyles, loc='upper left')
            ax1.tick_params(labelbottom=False);
            
        else:
            ax2.sharex(ax3)
            
        ax3.set_xlabel('Time [s]', labelpad=0)
        ax2.tick_params(labelbottom=False)
        
        if ('CmdDiff'.casefold() in obsType.casefold()) or\
            ('perftable'.casefold() in obsType.casefold()):
            gdBarLabel = 'Fringe Jumps\nPeriod [s]'
        else:
            gdBarLabel = 'GD rms\n[µm]'
        
        if not oneAxe:
            
            ct.setaxelim(ax4,ydata=np.concatenate([np.stack(RMSgdobs),[1]]),ymargin=0.2,ymin=0)
            ct.setaxelim(ax5,ydata=np.concatenate([np.stack(RMSpdobs)]),ymargin=0.2,ymin=0)
            ax4.bar_label(p1,label_type='edge',fmt='%.2f')
            ax5.bar_label(p3,label_type='edge',fmt='%.2f')
            ax9.bar_label(p2,label_type='edge',fmt='%.2f')
            ax10.bar_label(p4,label_type='edge',fmt='%.2f')
            # ax3.set_ylabel('Phase-Delays [µm]')
        
            ax5.set_xlabel('Baselines')
            
            ax4.tick_params(labelbottom=False,labelleft=False)
            ax5.tick_params(labelleft=False)
            
            ax4.get_shared_x_axes().join(ax4,ax5)
            ax9.get_shared_x_axes().join(ax9,ax10)
            
            ax2.get_shared_y_axes().join(ax2,ax7)
            ax3.get_shared_y_axes().join(ax3,ax8)
            ax4.get_shared_y_axes().join(ax4,ax9)
            ax5.get_shared_y_axes().join(ax5,ax10)

            ax7.tick_params(labelleft=False, labelbottom=False) ; ax8.tick_params(labelleft=False)
            ax9.tick_params(labelleft=False) ; ax10.tick_params(labelleft=False)
            
            ax9.tick_params(labelbottom=False)
            ax11.remove() ; ax12.remove()
            
            ax8.set_xlabel('Time [s]', labelpad=0)
            ax10.set_xlabel('Baselines')
            
            if plotSNR:
                ax6.get_shared_x_axes().join(ax6,ax7,ax8)
                ax1.get_shared_y_axes().join(ax1,ax6)
                ax6.tick_params(labelbottom=False, labelleft=False) ;
                
                ax7b=ax7.twinx() ; ax7b.set_ylabel('Group-Delays\n[µm]',
                                                   rotation=1,labelpad=50)
                ax8b=ax8.twinx() ; ax8b.set_ylabel('Phase-Delays\n[µm]',
                                                   rotation=1,labelpad=50)
                
                ct.setaxelim(ax7b,ydata=GDobs,ylim_min=[-wl/2,wl/2])
                ax8b.set_ylim([-wl/2,wl/2])
                ax9.grid(False); ax10.grid(False)
                ax9b = ax9.twinx() ; ax9b.set_ylabel(gdBarLabel,
                                                     rotation=1,labelpad=50)
                ct.setaxelim(ax9b,ydata=np.concatenate([np.stack(RMSgdobs),[1]]),
                             ymargin=0.2,ymin=0)
                
                ax10b = ax10.twinx() ; ax10b.set_ylabel('PD rms\n[µm]',
                                                        rotation=1,labelpad=50)#,loc='bottom')
                ct.setaxelim(ax10b,ydata=np.concatenate([np.stack(RMSpdobs)]),
                             ymargin=0.2,ymin=0)
                
                ax7b.sharey(ax2) ; ax8b.sharey(ax3)
                ax2.tick_params(labelleft=False)
                ax3.tick_params(labelleft=False)
                
            else:
                ax7.sharex(ax8)
                ax2.set_ylabel('Group-Delays [µm]')
                ax3.set_ylabel('Phase-Delays [µm]')
                
                ax4.set_ylabel(gdBarLabel,rotation=1,labelpad=60)#,loc='bottom')
                ax5.set_ylabel('PD rms\n[µm]',rotation=1,labelpad=60)#,loc='bottom')
                
                ax4.bar_label(p1,label_type='edge',fmt='%.2f')
                ax5.bar_label(p3,label_type='edge',fmt='%.2f')
                
                ct.setaxelim(ax2,ydata=GDobs,ylim_min=[-wl/2,wl/2],ymargin=0.2)
        
        else:
                
            ax2.set_ylabel('Group-Delays [µm]')
            ax3.set_ylabel('Phase-Delays [µm]')
            
            ct.setaxelim(ax4,xdata=np.concatenate([np.stack(RMSgdobs),[1]]),
                         ymargin=0.2,xmin=0)
            ct.setaxelim(ax5,xdata=np.concatenate([np.stack(RMSpdobs)]),
                         ymargin=0.2,xmin=0)
            ax4.bar_label(p1,label_type='edge',fmt='%.2f')
            ax5.bar_label(p3,label_type='edge',fmt='%.2f')
            ax4.set_xlabel(gdBarLabel)
            ax5.set_xlabel('PD rms\n[µm]')
            ax4.set_ylabel('Baselines',rotation=90,fontsize=14)
            ax5.tick_params(labelleft=False)
        
        
        if display:
            fig.show()

        if len(filename):
            if verbose:
                print("Saving perftable figure.")
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(filename+f"_{rangeCurves}.{extension}", dpi=300)
            else:
                plt.savefig(filename+f"_{rangeCurves}.{ext}", dpi=300)

    plt.rcParams.update(plt.rcParamsDefault)


def perftable_cp(timestamps, PDobs,GDobs,gdObsInfo,pdObsInfo,
                 plotObs,generalTitle,obsType,display=True,
                 filename='',ext='pdf',infos={"details":''},verbose=False):
    
    global closures,wl
                
    nObs = np.shape(PDobs)[1]
    nCurvesMaxPerFigure = 20
    nFigures = 1+nObs//nCurvesMaxPerFigure - 1*(nObs % nCurvesMaxPerFigure==0)
    curvesNames = closures
    
    plt.rcParams.update(rcParamsForOneAxe)

    nObsToPlot = np.sum(plotObs)
    plotObsIndex = np.argwhere(plotObs).ravel()

    # Each figure only shows 15 baselines, distributed on two subplots
    # If there are more than 15 baselines, multiple figures will be created
    for iFig in range(nFigures):
        nCurvesToDisplay=nCurvesMaxPerFigure
        if iFig == nFigures-1:
            nCurvesOnLastFigure = nObs%nCurvesMaxPerFigure
            if (nCurvesOnLastFigure < nCurvesMaxPerFigure) and (nCurvesOnLastFigure != 0):
                nCurvesToDisplay = nCurvesOnLastFigure
                
        iFirstCurve = nCurvesMaxPerFigure*iFig                # Index of first baseline to display
        iLastCurve = iFirstCurve + nCurvesToDisplay - 1       # Index of last baseline to display
        
        len2 = nCurvesToDisplay//2 ; len1 = nCurvesToDisplay-len2
        colorsArray = colors[:len1]+colors[:len2]
        colorsArray = np.array(colorsArray)
        
        oneAxe = False
        if nObsToPlot <=6:
            oneAxe=True

        if not oneAxe:
            rangeCurves = f"{curvesNames[iFirstCurve]}-{curvesNames[iLastCurve]}"
            title=f'{generalTitle}: {rangeCurves}'
        else:
            rangeCurves = ""
            title=generalTitle

        plt.close(title)
        fig=plt.figure(title, clear=True)
        fig.suptitle(title)
        
        if oneAxe:
            plt.rcParams.update(rcParamsForBaselines)
            ax2,ax3,ax11,ax4,ax5 = fig.subplots(nrows=5, gridspec_kw={"height_ratios":[4,4,0.7,1,1]})
            ax2.set_title(f"From {curvesNames[iFirstCurve]} to {curvesNames[iFirstCurve+len1-1]}")
        
        else:
            plt.rcParams.update(rcParamsForBaselines)
            (ax2,ax7), (ax3,ax8),(ax11,ax12),(ax4,ax9),(ax5,ax10) = fig.subplots(nrows=5,ncols=2, gridspec_kw={"height_ratios":[4,4,0.7,1,1]})
            ax2.set_title(f"From {curvesNames[iFirstCurve]} to {curvesNames[iFirstCurve+len1-1]}")
            ax7.set_title(f"From {curvesNames[iFirstCurve+len1]} to {curvesNames[iLastCurve]}")
        
        
        if oneAxe:
            colorsArrayOneAxe = colorsArray[:nObsToPlot]
            curvesNamesOneAxe = [curvesNames[iCurve] for iCurve in plotObsIndex]
            iColor=0
            for iCurve in plotObsIndex:   # First serie
                ax2.plot(timestamps,GDobs[:,iCurve],color=colorsArray[iColor])
                ax3.plot(timestamps,PDobs[:,iCurve],color=colorsArray[iColor])
                iColor+=1
                
            p1=ax4.bar(curvesNamesOneAxe,[gdObsInfo[iCurve] for iCurve in plotObsIndex], color=colorsArrayOneAxe)
            p3=ax5.bar(curvesNamesOneAxe,[pdObsInfo[iCurve] for iCurve in plotObsIndex], color=colorsArrayOneAxe)
            
        else:
            FirstSet = range(iFirstCurve,iFirstCurve+len1)
            SecondSet = range(iFirstCurve+len1,iLastCurve+1)
            iColor=0
            for iCurve in FirstSet:   # First serie
                ax2.plot(timestamps,GDobs[:,iCurve],color=colorsArray[iColor])
                ax3.plot(timestamps,PDobs[:,iCurve],color=colorsArray[iColor])
                iColor+=1
            for iCurve in SecondSet:   # Second serie
                ax7.plot(timestamps,GDobs[:,iCurve],color=colorsArray[iColor])
                ax8.plot(timestamps,PDobs[:,iCurve],color=colorsArray[iColor])
                iColor+=1
        
            p1=ax4.bar(curvesNames[FirstSet],gdObsInfo[FirstSet], color=colorsArray[:len1])
            p3=ax5.bar(curvesNames[FirstSet],pdObsInfo[FirstSet], color=colorsArray[:len1])
            
            p2=ax9.bar(curvesNames[SecondSet],gdObsInfo[SecondSet], color=colorsArray[len1:])
            p4=ax10.bar(curvesNames[SecondSet],pdObsInfo[SecondSet], color=colorsArray[len1:])
        
        """
        Tune the axis
        """
              
        ax2.sharex(ax3)
        ax2.set_ylabel('CP GD [°]',labelpad=-5)
        ax3.set_ylabel('CP PD [°]',labelpad=-5)
        
        ax3.set_xlabel('Time [s]', labelpad=0)
        ax2.tick_params(labelbottom=False)
        ax11.remove()
        
        ax4.set_ylabel('<CP GD>\n[°]', labelpad=10)
        ax5.set_ylabel('<CP PD>\n[°]', labelpad=10)
        
        ax4.bar_label(p1,label_type='edge',fmt='%.2f')
        ax5.bar_label(p3,label_type='edge',fmt='%.2f')
        ax5.tick_params(axis='x', labelrotation = 30)
        
        ct.setaxelim(ax4,ydata=np.concatenate([np.stack(gdObsInfo),[1]]),ymargin=0.2,ylim_min=[-5,+5])
        ct.setaxelim(ax5,ydata=np.concatenate([np.stack(pdObsInfo),[1]]),ymargin=0.2,ylim_min=[-5,+5])
        
        ax5.set_xlabel('Closures')
        
        if not oneAxe:
            ax12.remove()
            
            ax4.bar_label(p1,label_type='edge',fmt='%.2f')
            ax5.bar_label(p3,label_type='edge',fmt='%.2f')
            ax9.bar_label(p2,label_type='edge',fmt='%.2f')
            ax10.bar_label(p4,label_type='edge',fmt='%.2f')
            
            ax4.tick_params(labelbottom=False)
            
            ax4.get_shared_x_axes().join(ax4,ax5)
            ax9.get_shared_x_axes().join(ax9,ax10)
            
            ax2.get_shared_y_axes().join(ax2,ax7)
            ax3.get_shared_y_axes().join(ax3,ax8)
            ax4.get_shared_y_axes().join(ax4,ax9)
            ax5.get_shared_y_axes().join(ax5,ax10)

            ax7.tick_params(labelleft=False, labelbottom=False) ; ax8.tick_params(labelleft=False)
            ax9.tick_params(labelleft=False) ; ax10.tick_params(labelleft=False)
            
            ax9.tick_params(labelbottom=False)
            
            ax8.set_xlabel('Time [s]', labelpad=0)
            ax10.set_xlabel('Closures')
            ax10.tick_params(axis='x', labelrotation = 30)
            
            ax7.sharex(ax8)
            
            ct.setaxelim(ax2,ydata=GDobs,ylim_min=[-wl/2,wl/2],ymargin=0.2)
        
        
        if display:
            fig.show()

        if len(filename):
            if verbose:
                print("Saving perftable figure.")
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(filename+f"_{rangeCurves}.{extension}", dpi=300)
            else:
                plt.savefig(filename+f"_{rangeCurves}.{ext}", dpi=300)

    plt.rcParams.update(plt.rcParamsDefault)
    

def simpleplot_bases(timestamps, obs,obsBar,generalTitle,plotObs,
               obsName='PD [µm]',barName='RMS',display=True,filename='',ext='pdf',infos={"details":''},
               verbose=False):
    """
    Each figure only shows up to 15 baselines, distributed on two subplots
    If there are more than 15 baselines, multiple figures will be created 
    If there are less than 6 baselines, only one axe is plotted.
    In between, two axes on a unique figure are plotted.


    Parameters
    ----------
    timestamps : TYPE
        DESCRIPTION.
    obs : TYPE
        DESCRIPTION.
    obsBar : TYPE
        DESCRIPTION.
    generalTitle : TYPE
        DESCRIPTION.
    plotObs : TYPE
        DESCRIPTION.
    obsName : TYPE, optional
        DESCRIPTION. The default is 'PD [µm]'.
    display : TYPE, optional
        DESCRIPTION. The default is True.
    filename : TYPE, optional
        DESCRIPTION. The default is ''.
    ext : TYPE, optional
        DESCRIPTION. The default is 'pdf'.
    infos : TYPE, optional
        DESCRIPTION. The default is {"details":''}.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    global baselines,wl,NINdisp
    
    nObs = NINmes
    nCurvesMaxPerFigure = NINdisp
    nFigures = nBaseFigures
    curvesNames = baselines
    
    plt.rcParams.update(rcParamsForBaselines)

    nObsToPlot = np.sum(plotObs)            
    plotObsIndex = np.argwhere(plotObs).ravel()

    obsIsSnr=False
    if 'snr'.casefold() in obsName.casefold():
        obsIsSnr = True
        linestyles = [mlines.Line2D([],[],color='k',linestyle='--',label='Threshold GD')]
    
    for iFig in range(nFigures):
        nCurvesToDisplay=nCurvesMaxPerFigure
        if iFig == nFigures-1:
            nCurvesOnLastFigure = nObs%nCurvesMaxPerFigure
            if (nCurvesOnLastFigure < nCurvesMaxPerFigure) and (nCurvesOnLastFigure != 0):
                nCurvesToDisplay = nCurvesOnLastFigure
                
        iFirstCurve = nCurvesMaxPerFigure*iFig                       # Index of first baseline to display
        iLastBase = iFirstCurve + nCurvesToDisplay - 1       # Index of last baseline to display
        
        len2 = nCurvesToDisplay//2 ; len1 = nCurvesToDisplay-len2
        colorsArray = colors[:len1]+colors[:len2]
        colorsArray = np.array(colorsArray)
    
        oneAxe = False
        if nObsToPlot <= 6:
            oneAxe=True
    
        if not oneAxe:
            rangeCurves = f"{curvesNames[iFirstCurve]}-{curvesNames[iLastBase]}"
            title=f'{generalTitle}: {rangeCurves}'
        else:
            rangeCurves = ""
            title=generalTitle
        
        plt.close(title)
        fig=plt.figure(title, clear=True)
    
        if len(infos["details"]):
            fig.suptitle(title)
        

        if oneAxe:
            ax1,ax2 = fig.subplots(nrows=2,gridspec_kw={"height_ratios":[3,1]})
            colorsArrayOneAxe = colorsArray[:nObsToPlot]
            curvesNamesOneAxe = [curvesNames[iCurve] for iCurve in plotObsIndex]
            
            iColor=0
            for iCurve in plotObsIndex:
                line, = ax1.plot(timestamps,obs[:,iCurve],color=colorsArrayOneAxe[iColor],label=curvesNames[iCurve])
                # linestyles.append(line)
                
                if obsIsSnr:
                    ax1.hlines(config.FT['ThresholdGD'][iCurve], timestamps[0],timestamps[-1], 
                               color=colorsArray[iColor], linestyle='dashed')
                iColor+=1                
            
            p1=ax2.bar(curvesNamesOneAxe,[obsBar[iCurve] for iCurve in plotObsIndex], color=colorsArrayOneAxe)
            ax2.set_box_aspect(1/20)
    
        else:
            (ax1,ax3),(ax2,ax4) = fig.subplots(nrows=2,ncols=2, sharey='row',gridspec_kw={"height_ratios":[3,1]})
            ax1.set_title(f"From {curvesNames[iFirstCurve]} \
to {curvesNames[iFirstCurve+len1-1]}")
            ax3.set_title(f"From {curvesNames[iFirstCurve+len1]} \
to {curvesNames[iLastBase]}")
            
            FirstSet = range(iFirstCurve,iFirstCurve+len1)
            SecondSet = range(iFirstCurve+len1,iLastBase+1)
            NbOfObs = len(FirstSet)+len(SecondSet)
            barcolorsArray = ['grey']*NbOfObs
            
            iColor=0
            for iCurve in FirstSet:   # First serie
                if plotObs[iCurve]:
                    line, = ax1.plot(timestamps,obs[:,iCurve],color=colorsArray[iColor])
                    barcolorsArray[iColor] = colorsArray[iColor]
                    if obsIsSnr:
                        ax1.hlines(config.FT['ThresholdGD'][iCurve], timestamps[0],timestamps[-1],
                                            color=colorsArray[iColor], linestyle='dashed')

                iColor+=1
                
            for iCurve in SecondSet:   # Second serie
                if plotObs[iCurve]:
                    line, = ax3.plot(timestamps,obs[:,iCurve],color=colorsArray[iColor])
                    barcolorsArray[iColor] = colorsArray[iColor]
                    if obsIsSnr:
                        ax3.hlines(config.FT['ThresholdGD'][iCurve], timestamps[0],timestamps[-1],
                                            color=colorsArray[iColor], linestyle='dashed')

                iColor+=1
            
            p1=ax2.bar(curvesNames[FirstSet],obsBar[FirstSet], color=barcolorsArray[:len1])
            p2=ax4.bar(curvesNames[SecondSet],obsBar[SecondSet], color=barcolorsArray[len1:])
            # else:
            #     p1=ax2.bar(curvesNames[FirstSet],np.mean(obs[:,FirstSet],axis=0), color=barcolorsArray[:len1])
            #     p2=ax4.bar(curvesNames[SecondSet],np.mean(obs[:,SecondSet],axis=0), color=barcolorsArray[len1:])
                
            ax4.sharey(ax2) ; ax4.tick_params(labelleft=False)
            ct.setaxelim(ax1, ydata=obs, ymargin=0.4)
            if 'pd'.casefold() in obsName.casefold():
                ax4.set_ylim([0,wl])
            else:
                ct.setaxelim(ax4, ydata=obsBar,ymin=0)
            
            ax4.bar_label(p2,label_type='edge',fmt='%.2f')
            
            ax3.set_xlabel("Time [s]")
            ax4.set_xlabel("Baselines")
            ax4.set_anchor('S')
            ax2.set_box_aspect(1/6) ; ax4.set_box_aspect(1/6)
            
        if obsIsSnr:
            ct.setaxelim(ax1,ydata=[obs[:,iCurve] for iCurve in plotObsIndex],
                         ymargin=0.1,ymin=0,ylim_min=[0,1.2*np.max(config.FT['ThresholdGD'])])
            ax2.set_ylabel("Average")
            ct.setaxelim(ax2,ydata=list(obs), ymin=0,ymargin=0.1)
            ax1.legend(handles=linestyles)
        else:
            ct.setaxelim(ax1,ydata=[obs[:,iCurve] for iCurve in plotObsIndex],
                         ymargin=0.1)
            ax2.set_ylabel(barName)
            ct.setaxelim(ax2,ydata=list(obsBar)+[wl/2], ymin=0)
            
        ax1.set_ylabel(obsName)
        ax1.set_xlabel("Time [s]") ;             
        ax2.set_xlabel("Baselines") ; 
        ax2.set_anchor('S')
        
        ax2.bar_label(p1,label_type='edge',fmt='%.2f')
    
        if display:
            fig.show()
            
        if len(filename):
            if verbose:
                print("Saving perftable figure.")
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(filename+f"_{rangeCurves}.{extension}", dpi=300)
            else:
                plt.savefig(filename+f"_{rangeCurves}.{ext}", dpi=300)                
                
def simpleplot_tels(timestamps, obs,obsBar,generalTitle,plotObs,mov_average=0,
                    obsName='PD [µm]',barName='RMS',display=True,filename='',ext='pdf',infos={"details":''},
                    verbose=False):
    """
    Each figure only shows up to 10 telescopes, distributed on two subplots
    If there are more than 10 telescopes, multiple figures will be created 
    If there are less than 6 telescopes, only one axe is plotted.
    In between, two axes on a unique figure are plotted.


    Parameters
    ----------
    timestamps : TYPE
        DESCRIPTION.
    obs : TYPE
        DESCRIPTION.
    obsBar : TYPE
        DESCRIPTION.
    generalTitle : TYPE
        DESCRIPTION.
    plotObs : TYPE
        DESCRIPTION.
    obsName : TYPE, optional
        DESCRIPTION. The default is 'PD [µm]'.
    display : TYPE, optional
        DESCRIPTION. The default is True.
    filename : TYPE, optional
        DESCRIPTION. The default is ''.
    ext : TYPE, optional
        DESCRIPTION. The default is 'pdf'.
    infos : TYPE, optional
        DESCRIPTION. The default is {"details":''}.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    global telescopes, wl,NAdisp
    
    nObs = NA
    nCurvesMaxPerFigure = NAdisp
    nFigures = nTelFigures
    
    if mov_average:
        obs = ct.moving_average(obs, mov_average)
        timestamps = ct.moving_average(timestamps, mov_average)
        
    plt.rcParams.update(rcParamsForBaselines)
    
    # Each figure only shows up to 10 telescopes, distributed on two subplots
    # If there are more than 10 baselines, multiple figures will be created 
    # If there are less than 6 baselines, only one axe is plotted.
    # In between, two axes on a unique figure are plotted.
    
    obsIsSnr=False ; obsIsFlux=False
    if 'singularValuesSqrt'.casefold() in obsName.casefold():
        obsIsSnr = True
        linestyles = [mlines.Line2D([],[],color='k',linestyle='--',label='Threshold PD')]
        plotObs = [True]*np.shape(obs)[1]
        
    if 'flux'.casefold() in obsName.casefold():
        obsIsFlux = True
    
    nObsToPlot = np.sum(plotObs)
    plotObsIndex = np.argwhere(plotObs).ravel()
    
    for iFig in range(nFigures):
        nCurvesToDisplay=nCurvesMaxPerFigure
        if iFig == nFigures-1:
            nCurvesOnLastFigure = nObs%nCurvesMaxPerFigure
            if (nCurvesOnLastFigure < nCurvesMaxPerFigure) and (nCurvesOnLastFigure != 0):
                nCurvesToDisplay = nCurvesOnLastFigure
                
        iFirstCurve= nCurvesMaxPerFigure*iFig                         # Index of first baseline to display
        iLastTel= iFirstCurve + nCurvesToDisplay - 1          # Index of last baseline to display
        
        oneAxe = False
        if nObsToPlot <= 6:
            oneAxe=True
            
        if not oneAxe:
            rangeTels= f"{telescopes[iFirstCurve]}-{telescopes[iLastTel]}"
            title=f'{generalTitle}: {rangeTels}'
        else:
            rangeTels=""
            title=generalTitle
            
        plt.close(title)
        fig=plt.figure(title, clear=True)
    
        if len(infos["details"]):
            fig.suptitle(title)

        if oneAxe:
            telcolors = colors[:nObsToPlot]
            ax1,ax2 = fig.subplots(nrows=2,gridspec_kw={"height_ratios":[3,1]})
            telcolorstemp = telcolors[:nObsToPlot]
            
            if obsIsSnr:
                telescopestemp = [[f"$s_{i}$" for i in np.arange(1,NA+1)][iCurve] for iCurve in plotObsIndex]
            else:
                telescopestemp = [telescopes[iCurve] for iCurve in plotObsIndex]    
            
            iColor=0
            for iCurve in plotObsIndex:
                ax1.plot(timestamps,obs[:,iCurve],color=telcolorstemp[iColor],label=telescopes[iCurve])
                iColor+=1           
            
            if obsIsSnr:
                ax1.hlines(config.FT['ThresholdPD'], timestamps[0],timestamps[-1],
                           color="k", linestyle='dashed')

            p1=ax2.bar(telescopestemp,[obsBar[iCurve] for iCurve in plotObsIndex], color=telcolorstemp)
            
            ax2.set_box_aspect(1/20)
            ax1.legend()
    
        else:
            len2 = nCurvesToDisplay//2 ; len1 = nCurvesToDisplay-len2
            telcolors = colors[:len1]+colors[:len2]
            telcolors = np.array(telcolors)
            (ax1,ax3),(ax2,ax4) = fig.subplots(nrows=2,ncols=2, sharey='row',gridspec_kw={"height_ratios":[3,1]})
            ax1.set_title(f"From {telescopes[iFirstCurve]} \
to {telescopes[iFirstCurve+len1-1]}")
            ax3.set_title(f"From {telescopes[iFirstCurve+len1]} \
to {telescopes[iLastTel]}")
            
            FirstSet = range(iFirstCurve,iFirstCurve+len1)
            SecondSet = range(iFirstCurve+len1,iLastTel+1)
            NbOfObs = len(FirstSet)+len(SecondSet)
            barcolors = ['grey']*NbOfObs
            
            iColor=0
            for iCurve in FirstSet:   # First serie
                if plotObs[iCurve]:
                    ax1.plot(timestamps,obs[:,iCurve],color=telcolors[iColor])
                    barcolors[iColor] = telcolors[iColor]
                iColor+=1
                
            for iCurve in SecondSet:   # Second serie
                if plotObs[iCurve]:
                    ax3.plot(timestamps,obs[:,iCurve],color=telcolors[iColor])
                    barcolors[iColor] = telcolors[iColor]
                iColor+=1
             
            if obsIsSnr:
                ax1.hlines(config.FT['ThresholdPD'], timestamps[0],timestamps[-1],
                           color="k", linestyle='dashed')
                ax2.hlines(config.FT['ThresholdPD'], timestamps[0],timestamps[-1],
                           color="k", linestyle='dashed')
                
                p1=ax2.bar([f"$s_{i}$" for i in np.arange(1,NA+1)][FirstSet],np.mean(obs[FirstSet],axis=0), color=barcolors[:len1])
                p2=ax4.bar([f"$s_{i}$" for i in np.arange(1,NA+1)][SecondSet],np.mean(obs[SecondSet],axis=0), color=barcolors[len1:])
            else:
                p1=ax2.bar(telescopes[FirstSet],obsBar[FirstSet], color=barcolors[:len1])
                p2=ax4.bar(telescopes[SecondSet],obsBar[SecondSet], color=barcolors[len1:])

            ax4.sharey(ax2) ; ax4.tick_params(labelleft=False)
            ct.setaxelim(ax1, ydata=obs, ymargin=0.4,ymin=0)
            if 'pd'.casefold() in obsName.casefold():
                ax4.set_ylim([0,wl])
            else:
                ct.setaxelim(ax4, ydata=obsBar,ymin=0)
            
            ax4.bar_label(p2,label_type='edge',fmt='%.2f')
            
            ax3.set_xlabel("Time [s]")
            ax4.set_xlabel("Telescopes")
            ax4.set_anchor('S')
            ax2.set_box_aspect(1/6) ; ax4.set_box_aspect(1/6)
            
            
        # ct.setaxelim(ax1,ydata=[obs[:,iCurve] for iCurve in plotObsIndex])
        # ct.setaxelim(ax2,ydata=list(obsBar)+[wl/2], ymin=0)
        
        if obsIsSnr or obsIsFlux:
            ax2.set_ylabel("Average")
            ct.setaxelim(ax2,ydata=list(obs), ymin=0,ymargin=0.1)
            
            if obsIsSnr:
                ct.setaxelim(ax1,ydata=[obs[:,iCurve] for iCurve in plotObsIndex],
                             ymargin=0.2,ymin=0,ylim_min=[0,1.1*config.FT['ThresholdPD']])
                ax2.set_xlabel("Singular values")
                ax1.legend(handles=linestyles)
            else:
                ct.setaxelim(ax1,ydata=[obs[:,iCurve] for iCurve in plotObsIndex],
                             ymargin=0.2,ymin=0)
                ax2.set_xlabel("Telescopes")
                
        else:
            ct.setaxelim(ax1,ydata=[obs[:,iCurve] for iCurve in plotObsIndex],
                         ymargin=0.2)
            ax2.set_ylabel(barName)
            ct.setaxelim(ax2,ydata=list(obsBar)+[wl/2], ymin=0)
            ax2.set_xlabel("Telescopes") ; 
        
        ax1.set_xlabel("Time [s]") ; 
        ax1.set_ylabel(obsName)
        ax2.set_anchor('S')
        
        ax2.bar_label(p1,label_type='edge',fmt='%.2f')
    
        if display:
            fig.show()
            
        if len(filename):
            if verbose:
                print("Saving perftable figure.")
            if isinstance(ext,list):
                for extension in ext:
                    plt.savefig(filename+f"_{rangeTels}.{extension}", dpi=300)
            else:
                plt.savefig(filename+f"_{rangeTels}.{ext}", dpi=300)


# def simpleplot_cp(timestamps, obs,obsBar,generalTitle,plotObs,
#                 obsName='PD [µm]',barName='RMS',display=True,filename='',ext='pdf',infos={"details":''},
#                 verbose=False):


#     plt.rcParams.update(rcParamsForBaselines)
#     title=f'Closure phases - {details}'
#     plt.close(title)
#     fig=plt.figure(title, clear=True)
#     fig.suptitle(title)
#     (ax1,ax5),(ax2,ax6),(ax11,ax12), (ax3,ax7) = fig.subplots(nrows=4,ncols=2, gridspec_kw={"height_ratios":[5,5,1,2]})
#     ax1.set_title("First serie of closure phases")
#     ax5.set_title("Second serie of closure phases")

#     len2cp = NC//2 ; len1cp=NC-len2cp
#     for iClosure in range(len1cp):
#         if PlotClosure[iClosure]:
#             # ax1.plot(t[timerange],1/averagePdVar[timerange,iClosure],color=colorsArray[iClosure])
#             ax1.plot(t[timerange],gdClosure[timerange,iClosure]*R*180/np.pi,color=colorsArray[iClosure])
#             ax2.plot(t[timerange],pdClosure[timerange,iClosure]*180/np.pi,color=colorsArray[iClosure])

#         ax3.errorbar(iClosure,Meangdc[iClosure]*R*180/np.pi,yerr=RMSgdc[iClosure]*R*180/np.pi, marker='o',color=colorsArray[iClosure],ecolor='k')
#         ax3.errorbar(iClosure,Meanpdc[iClosure]*180/np.pi,yerr=RMSpdc[iClosure]*180/np.pi, marker='x',color=colorsArray[iClosure],ecolor='k')

#     for iClosure in range(len1cp,NC):
#         if PlotClosure[iClosure]:
#             # ax6.plot(t[timerange],1/averagePdVar[timerange,iClosure],color=colorsArray[iClosure])
#             ax5.plot(t[timerange],gdClosure[timerange,iClosure]*R*180/np.pi,color=colorsArray[iClosure])
#             ax6.plot(t[timerange],pdClosure[timerange,iClosure]*180/np.pi,color=colorsArray[iClosure])

#         ax7.errorbar(iClosure,Meangdc[iClosure]*R*180/np.pi,yerr=RMSgdc[iClosure]*R*180/np.pi, marker='o',color=colorsArray[iClosure],ecolor='k')
#         ax7.errorbar(iClosure,Meanpdc[iClosure]*180/np.pi,yerr=RMSpdc[iClosure]*180/np.pi, marker='x',color=colorsArray[iClosure],ecolor='k')

#     ax1.sharex(ax2) ; ax5.sharex(ax6)
#     ax5.sharey(ax1) ; ax5.tick_params(labelleft=False) ; setaxelim(ax1,ydata=gdClosure[timerange[200:]]*R*180/np.pi)
#     ax6.sharey(ax2) ; ax6.tick_params(labelleft=False) ; setaxelim(ax2,ydata=pdClosure[timerange[200:]]*180/np.pi)
#     ax7.sharey(ax3) ; ax7.tick_params(labelleft=False) ; setaxelim(ax3,ydata=Meangdc*R*180/np.pi,absmargin=True)

#     ax3.set_xticks(np.arange(len1cp))
#     ax7.set_xticks(np.arange(len1cp,NC))
#     ax3.set_xticklabels(closures[:len1cp],rotation=45)
#     ax7.set_xticklabels(closures[len1cp:],rotation=45)

#     ax3.sharey(ax7); ax3.set_ylim(-270,270)
#     ax3.set_yticks([-180,-90,0,90,180])
#     ax3.set_yticklabels([-180,'',0,'',180])
#     #ax7.set_yticks([-180,-90,0,90,180])
    
#     ax1.set_ylabel('GD closure [°]')
#     ax2.set_ylabel('PD closure [°]')
#     ax3.set_ylabel('Mean \nvalues\n[°]',rotation=90,labelpad=10,loc='bottom')
#     ax1.legend() ; ax5.legend()
#     ax11.remove() ; ax12.remove() # These axes are here to let space for ax3 and ax8 labels

#     ax2.set_xlabel("Time (s)") ; ax6.set_xlabel("Time (s)")

#     figname = '_'.join(title.split(' ')[:3])
#     figname = 'GD&PDcp'
#     if figsave:
#         if isinstance(ext,list):
#             for extension in ext:
#                 plt.savefig(figdir+f"{prefix}_{figname}.{extension}")
#         else:
#             plt.savefig(figdir+f"{prefix}_{figname}.{ext}")
#     fig.show()

def plotHisto(obs,generalTitle,plotObs,obsName='GD [µm]',
              display=True,filename='',ext='pdf',infos={"details":''},
              verbose=False):
    
    # obsIsGd=False ; obsIsFlux=False
    if "gd".casefold() in obsName.casefold():
        # obsIsGd = True
        curvesNames = baselines
        rangeX,rangeY = -2.5*wl,2.5*wl
    elif "flux".casefold() in obsName.casefold():
        # obsIsFlux=True
        curvesNames = telescopes
        rangeX,rangeY = np.min(obs),np.max(obs)*1.1
    
    plt.close(generalTitle)
    fig=plt.figure(generalTitle, clear=True)
    fig.suptitle(generalTitle)
    
    if np.sum(plotObs) == 0:        # Put to 0 means it nevers enter this loop.
        ax = fig.subplots()
        plottedCurve = f"_{curvesNames[plotObs][0]}"
        
        iCurve = np.argwhere(plotObs)[0][0]
        p=ax.hist(obs[:,iCurve], bins=100, range=(rangeX,rangeY),color='k')
        pmax = np.max(p[0])
            
        ax.set_ylabel("Occurences")
        ax.set_xlabel(obsName)
        
        ax.annotate(curvesNames[iCurve],(-4,0.9*pmax))
        
        ax.set_ylim(0,1.1*pmax)
        ax.set_xlim(rangeX,rangeY)
        
        
    else:
        plottedCurve = ""
        nCurves = np.shape(obs)[1]
        nrows=int(np.sqrt(nCurves))# ; nrows = val+1
        # minimalDivider = int(np.sqrt(nCurves)) ; nrows = minimalDivider
        # for divider in range(minimalDivider,nCurves//minimalDivider+1):
        #     if nCurves//divider > minimalDivider:
        #         nrows = divider
        otherAxe = nCurves//nrows ; remainder = nCurves%nrows
        if remainder:
            ncols = otherAxe+2
        else:
            ncols=otherAxe
        axes = fig.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True)
        
        if remainder:
            for iAxe in range(1,nrows-remainder+1):
                axes[-1,-iAxe].remove()
        axes = axes.ravel()
        
        pmax=0
        for iCurve in range(nCurves):   # First serie
            p=axes[iCurve].hist(obs[:,iCurve], bins=100, range=(rangeX,rangeY),color='k')
            if np.max(p[0]) > pmax:
                pmax = np.max(p[0])
                
            if iCurve%ncols == 0:
                axes[iCurve].set_ylabel("Occurences")
                
            if iCurve > nrows*(ncols-1)-1:
                axes[iCurve].set_xlabel(obsName)
            
        for iCurve in range(nCurves):
            addtext(axes[iCurve],curvesNames[iCurve],loc="upper right",fontsize='x-large')
        
        axes[0].set_ylim(0,1.1*pmax)
        axes[0].set_xlim(rangeX,rangeY)

    if display:
        fig.show()
        
    if len(filename):
        
        if verbose:
            print("Saving histogram figure.")
        if isinstance(ext,list):
            for extension in ext:
                plt.savefig(filename+f"{plottedCurve}.{extension}", dpi=300)
        else:
            plt.savefig(filename+f"{plottedCurve}.{ext}", dpi=300)




def axPerfarray(ax,lsObs,colorObs,axTitle,lwObs=[]):
    
    for ia in range(NA):
        name1,(x1,y1) = telescopes[ia],InterfArray.TelCoordinates[ia,:2]
        ax.scatter(x1,y1,color='k',linewidth=10)
        ax.annotate(name1, (x1+6,y1+1),color="k")
        ax.annotate(f"({ia+1})", (x1+21,y1+1),color=colors[0])
        for iap in range(ia+1,NA):
            ib=ct.posk(ia,iap,NA)
            x2,y2 = InterfArray.TelCoordinates[iap,:2]
            im=ax1.plot([x1,x2],[y1,y2],linestyle='solid',
                    linewidth=1,
                    color=cm(int(vismod[ib]*nShades)))
    ax.set_xlabel("X [m]")
    ax.tick_params(labelleft=False)
    
    for ia in range(NA):
        name1,(x1,y1) = InterfArray.TelNames[ia],InterfArray.TelCoordinates[ia,:2]
        ax2.scatter(x1,y1,color='k',linewidth=10)
        ax2.annotate(name1, (x1+6,y1+1),color="k")
        ax2.annotate(f"({ia+1})", (x1+21,y1+1),color=colors[0])
        for iap in range(ia+1,NA):
            ib=ct.posk(ia,iap,NA)
            x2,y2 = InterfArray.TelCoordinates[iap,:2]
            ls = (0,(10*lsObs[ib],np.max([0,10*(1-lsObs[ib])])))
            if PhotometricBalance[ib]>0:
                im=ax2.plot([x1,x2],[y1,y2],linestyle=ls,
                        linewidth=3,
                        color=cm(int(outputs.FringeContrast[ib]*nShades)))
            else:
                im=ax2.plot([x1,x2],[y1,y2],linestyle=ls,
                        linewidth=1,
                        color=cm(int(outputs.FringeContrast[ib]*nShades)))
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_xlim([-210,160]) ; ax2.set_ylim([-50,350])



def perfarray(lsObs,colorObs,generalTitle,lwObs1=[],axTitle="Standard deviation",
              lsObs2=[],colorObs2=[],lwObs2=[],axTitle2="Standard deviation",
              infos={}):
    
    plt.rcParams.update(rcParamsForBaselines)
    
    from .tol_colors import tol_cmap as tc
    import matplotlib as mpl
    from .config import NA,NIN,InterfArray,wlOfTrack
    
    twoAxes=False
    if len(lsObs2):
        twoAxes = True
        
    #visibilities, _,_,_=ct.VanCittert(wlOfScience,config.Obs,config.Target)
    #outputs.VisibilityAtPerfWL = visibilities
    # visibilities = np.ones(NIN)  #ct.NB2NIN(outputs.VisibilityObject[wlIndex])
    # vismod = np.abs(visibilities) ; #visangle = np.angle(visibilities)
    # PhotometricBalance = config.FS['PhotometricBalance']
    
    cm = tc('rainbow_PuRd').reversed() ; nShades = 256
    
    # plt.rcParams['figure.figsize']=(16,12)
    # font = {'family' : 'DejaVu Sans',
    #         'weight' : 'normal',
    #         'size'   : 22}
    
    # plt.rc('font', **font)
    plt.close(generalTitle)
    fig=plt.figure(generalTitle)
    if twoAxes:
        (ax1,ax2)=fig.subplots(ncols=2, sharex=True, sharey=True)
        ax2.set_title(axTitle2)
    else:
        ax1 = fig.subplots()
    
    ax1.set_title(axTitle)
    
    axPerfarray(lsObs,colorObs,axTitle,lsObs=lsObs)
    
    if twoAxes:
        axPerfarray(lsObs2,colorObs2,axTitle2,lsObs=lsObs2)
    
    

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # mpl.colorbar.ColorbarBase(cbar_ax, cmap=cm,
    #                           orientation='vertical',
    #                           label=f"Fringe Contrast at {wlOfScience:.3}µm")


    
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.1, 0.05, 0.85, 0.05])
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cm,
                              orientation='horizontal')

    if len(savedir):
        if verbose:
            print("Saving perfarray figure.")
        plt.savefig(savedir+f"{filenamePrefix}_perfarray.{ext}")

    plt.rcParams.update(plt.rcParamsDefault)


def addtext(ax, text, loc = 'best', fontsize='small',fancybox=True, 
            framealpha=0.7, handlelength=0, handletextpad=0):
    """
    Add a text in a legend box. Enables to use the very useful "loc" parameter 
    of the built-in legend function.

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    text : TYPE
        DESCRIPTION.
    loc : TYPE, optional
        DESCRIPTION. The default is 'best'.
    fontsize : TYPE, optional
        DESCRIPTION. The default is 'small'.
    fancybox : TYPE, optional
        DESCRIPTION. The default is True.
    framealpha : TYPE, optional
        DESCRIPTION. The default is 0.7.
    handlelength : TYPE, optional
        DESCRIPTION. The default is 0.
    handletextpad : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    import matplotlib.patches as mpl_patches
    # create a list with two empty handles (or more if needed)
        
    n = 20
    textList = [text[i:i+n] for i in range(0, len(text), n)]
    Nlines = len(textList)
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", 
                                     lw=0, alpha=0)] * Nlines

    labels = []
    for texttemp in textList:
        labels.append(texttemp)
    
    # create the legend, supressing the blank space of the empty line symbol and the
    # padding between symbol and label by setting handlelenght and handletextpad
    ax.legend(handles, labels, loc=loc, fontsize='small', 
              fancybox=True, framealpha=0.7, 
              handlelength=0, handletextpad=0)