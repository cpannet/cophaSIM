"""
Creation: 2023-10-24
Author: cpannetier
Contact: cyril.pannetier@hotmail.fr
            
"""

import os, sys, glob
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits

from cophasim import TrueTelemetries as TT
from cophasim import skeleton as sk
from cophasim import coh_tools as ct
from cophasim import config,outputs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def writeInFits(filepath,fileHdr, hdr_dico, df_bases,df_tels):
    """
    Modify fits file to add header and performance records.

    Parameters
    ----------
    
    filepath : STRING
        Path of the fits file to create.
    fileHdr : FITS header
        Header of the PrimaryHDU on which adding hdr_dico items.
    hdr_dico : DICTIONNARY
        Dictionnary with keys and values to put in the header.
    df_bases : DICTIONNARY
        Dictionnary with keys and values concerning baselines performance.
    df_tels : DICTIONNARY
        Dictionnary with keys and values concerning telescopes performance.

    Returns
    -------
    None.

    """
    
    # Create the PrimaryHDU and add items to the heander
    hdu0 = fits.PrimaryHDU() ; hdu0.header = fileHdr
    
    fileHdr["Kpd"] = (hdr_dico["Kpd"],"Gain PD")
    fileHdr["Kgd"] = (hdr_dico["Kgd"],"Gain GD")
    fileHdr["Spd"] = (hdr_dico["Spd"],"Threshold PD")
    fileHdr["start"] = (hdr_dico["timebonds"][0],"Start of sequence, in seconds, on which perf are computed.")
    fileHdr["stop"] = (hdr_dico["timebonds"][1],"End of sequence, in seconds, on which perf are computed.")
    fileHdr['nTrackedBaselines'] = (hdr_dico["nTrackedBaselines"],"Number of tracked baselines (SNR>Sgd, 80% of the time)")
    fileHdr['DIT in science'] = (hdr_dico["DIT"],"DIT for computing the rms and other performance quantities")
    fileHdr['min Rms Pd'] = (hdr_dico["minRmsPd"],"Best rms(PD) in microns among tracked baselines")
    fileHdr['max Rms Pd'] = (hdr_dico["maxRmsPd"], "Worst rms(PD) in microns among tracked baselines")
    fileHdr['median Rms Pd'] = (hdr_dico["medianRmsPd"],"Median rms(PD) in microns among tracked baselines")
    fileHdr['median Rms Disp'] = (hdr_dico["medianRmsDisp"], "Median rms(disp) in microns among tracked baselines")
    fileHdr['min Gamma'] = (hdr_dico["minGamma"], "Minimal contrast loss in science instrument among tracked baselines, due to OPD residuals")
    fileHdr['median Gamma'] = (hdr_dico["medianGamma"], "Median contrast loss in science instrument among tracked baselines, due to OPD residuals")
    fileHdr['median T0'] = (hdr_dico["medianT0"], "Median t0 in ms among tracked baselines")
    fileHdr['median FJP'] = (hdr_dico["medianFjpBases"], "Median FJP in seconds among tracked baselines")
    fileHdr['nInjectedTels'] = (hdr_dico["nInjectedTels"],"Number of injected tels (flux >0, 80% of the time)")
    fileHdr['min Flux'] = (hdr_dico["minFlux"],"Lowest median flux among injected tels")
    fileHdr['max Flux'] = (hdr_dico["maxFlux"], "Highest median flux among injected tels")
    fileHdr['median Flux'] = (hdr_dico["medianFluxTels"], "Median flux among injected tels")
    
    
    # Create the secondary HDUs for baselines and telescopes performance
    hdr = fits.Header()
    
    colsList=[]
    for key,value in df_bases.items():
        if "°" in key:
            key=key.replace("°","deg")
        if isinstance(value[0],float):
            valFormat = "D"
        elif isinstance(value[0],str):
            valFormat = f"{len(value[0])}A"
        elif isinstance(value[0],np.bool_):
            valFormat = "L"
        colsList.append(fits.Column(name=key,format=valFormat,array=value))
    
    cols = fits.ColDefs(colsList)
    hduBase = fits.BinTableHDU.from_columns(cols,header=hdr)
    hduBase.name = "bases performance"
    
    colsList=[]
    for key,value in df_tels.items():
        if isinstance(value[0],float):
            valFormat = "D"
        elif isinstance(value[0],str):
            valFormat = f"{len(value[0])}A"
        elif isinstance(value[0],np.bool_):
            valFormat = "L"
        colsList.append(fits.Column(name=key,format=valFormat,array=value))
    
    cols = fits.ColDefs(colsList)
    
    hduTel = fits.BinTableHDU.from_columns(cols,header=hdr)
    hduTel.name = "tels performance"

    newhduL = fits.HDUList([hdu0,hduBase,hduTel])
    newhduL.writeto(filepath)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='showMainPlots',
        description="Load spica-ft telemetries and:\n\
            - shows the main figures caracterising performance\n\
            - Adds header and performance results to the fits file.\n\
            - optionally:\n\
                - shows and saves specific quantities: see 'display' function \
help for a detail of which data can be plotted.")

    parser.add_argument(
        'filesOrDir',
        default="",
        help='A file, a list of files, or a directory.'
        )
    parser.add_argument(
        '--timebonds',
        default=(0,-1),nargs='*',type=float,
        help='Plot only the sequence within the time bonds.',
        action='store'
        )
    parser.add_argument(
        '--dit',
        default=1,nargs='*',type=float,
        help='DIT (in seconds) on which standard deviation is computed. Default is 1 second.',
        action='store'
        )
    parser.add_argument(
        '--show',
        default=[],nargs='*',type=str,
        help='Show specific quantities by calling them.',
        action='store'
        )
    parser.add_argument(
        '--ext',
        default=0,nargs='*',type=str,
        help='List all extensions you want the figures to be saved to. Pdf by default.',
        action='store'
        )
    parser.add_argument(
        '--no-mergedpdf',
        default=0,
        help="Don't save figures in a common pdf file. Default is False.",
        action='store_true'
        )
    parser.add_argument(
        '--display',
        default=0,
        help='Display the figures. Default is False.',
        action='store_true'
        )
    parser.add_argument(
        '--perf',
        default=0,
        help='Compute and save main performance indicator in a fitsfile.',
        action='store_true'
        )
    parser.add_argument(
        '--print',
        default=0,
        help='Print main performances in line.',
        action='store_true'
        )
    parser.add_argument(
        '--wlOfScience',
        default=0.75,nargs='*',
        help='Wavelength of the science instrument to compute visibility loss.',
        action='store'
        )
    
    
    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning showMainPlots.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        print("\n     Example : python showMainPlots.py . --perf")
        sys.exit(0)

    InterfArray = config.Interferometer()
    InterfArray.get_array(name='chara')
    
    filesOrDir = args.filesOrDir
    dir0=os.getcwd()
    if os.path.isdir(filesOrDir):
        os.chdir(filesOrDir)
        filesOrDir = glob.glob(dir0+"/*.fits")
    else:
        filesOrDir=[filesOrDir]

    timebonds = tuple(args.timebonds)
    dit = args.dit
    
    dataToShow = args.show
    outputsData=[]; specialData=[]
    for dataName in dataToShow:
        if dataName in vars(outputs).keys():
            outputsData.append(dataName)
        else:
            specialData.append(dataName)
            
    mergedPdf = not args.no_mergedpdf

    wlOfScience = args.wlOfScience
    
    for filepath in filesOrDir:

        TT.ReadFits(filepath)
        sk.display(*specialData, outputsData=outputsData, timebonds=timebonds, DIT=dit,mergedPdf=mergedPdf,
                   savedir=dir0+'/figures/',display=args.display)
        
        if args.perf:
                    
            sk.ShowPerformance(timebonds, wlOfScience, dit, display=False)
            freqs,atmPsd,freqsSmoothed,atmPsdSmoothed = ct.getPsd(outputs.OPDCommand,outputs.timestamps,timebonds=timebonds)
            t0,coefs = ct.estimateT0(freqsSmoothed,atmPsdSmoothed,f2=2)
            np.nan_to_num(t0,posinf=0,neginf=0,copy=False)
            
            medianFlux = np.median(outputs.PhotometryEstimated,axis=0)
            injectedRatio = np.sum(outputs.PhotometryEstimated > 0, axis=0)/config.NT
            injectedTelsBoolean = injectedRatio > 0.8
            injectedTels = InterfArray.TelNames[injectedTelsBoolean]
    
            trackedRatio = np.sum(outputs.SquaredSNRMovingAveragePD > config.FT['ThresholdGD'],axis=0)/config.NT
            trackedBaselinesBoolean = trackedRatio > 0.8
            trackedBaselines = InterfArray.BaseNames[trackedBaselinesBoolean]
            baseAzimuth = np.arctan(InterfArray.BaseCoordinates[:,0]/InterfArray.BaseCoordinates[:,1])*180/np.pi
    
            df_bases = pd.DataFrame({"baseline":InterfArray.BaseNames,"length [m]":InterfArray.BaseNorms,"azimuth [°]":baseAzimuth,"tracked":trackedBaselinesBoolean,"trackedRatio":trackedRatio,
                              "t0 [ms]":t0*1e3,"rms(PD) [micron]":np.sqrt(outputs.VarPDEst)*config.wlOfTrack/2/np.pi,"FJP [s]":outputs.fringeJumpsPeriod,
                             "rms(GD) [micron]":np.sqrt(outputs.VarGDEst)*config.FS['R']*config.wlOfTrack/2/np.pi,
                              "rms(disp) [micron]":np.sqrt(outputs.VarDispersion),"mean(SNR uncoh)":np.sqrt(np.mean(outputs.SquaredSnrGD,axis=0)),
                             "mean(SNR coh)":np.sqrt(np.mean(outputs.SquaredSnrPD,axis=0)),
                                    "Sgd":np.round(config.FT['ThresholdGD'],2),"FC in visible":outputs.FringeContrast})
    
            df_tels = pd.DataFrame({"tel":InterfArray.TelNames,"injected":injectedTelsBoolean,"median Flux":medianFlux, "injectedRatio":injectedRatio, "FJP [s]":outputs.fringeJumpsPeriodTel})
    
            nTrackedBaselines = np.sum(trackedBaselinesBoolean)
            minRmsPd = round(df_bases[trackedBaselinesBoolean]["rms(PD) [micron]"].min(),2)
            maxRmsPd = round(df_bases[trackedBaselinesBoolean]["rms(PD) [micron]"].max(),2)
            medianRmsPd = round(df_bases[trackedBaselinesBoolean]["rms(PD) [micron]"].median(),2)
            medianRmsDisp = round(df_bases[trackedBaselinesBoolean]["rms(disp) [micron]"].median(),2)
            medianT0 = round(df_bases[trackedBaselinesBoolean]["t0 [ms]"].median(),0)
            medianFjpBases = round(df_bases[trackedBaselinesBoolean]["FJP [s]"].median(),2)
            minFC = round(df_bases[trackedBaselinesBoolean]["FC in visible"].min(),2)
            maxFC = round(df_bases[trackedBaselinesBoolean]["FC in visible"].max(),2)
            medianFC = round(df_bases[trackedBaselinesBoolean]["FC in visible"].median(),2)
            minGamma = round(1-maxFC,2) ; maxGamma = round(1-minFC,2) ; medianGamma = round(1-medianFC,2)

            nInjectedTels = np.sum(injectedTelsBoolean)
            minFlux = round(df_tels[injectedTelsBoolean]["median Flux"].min(),0)
            maxFlux = round(df_tels[injectedTelsBoolean]["median Flux"].max(),0)
            medianFluxTels = round(df_tels[injectedTelsBoolean]["median Flux"].median(),0)
            
            hdr_dico = {"timebonds":timebonds,"Kpd":round(config.FT['GainPD'],2),
                     "Kgd":round(config.FT['GainGD'],2),"Spd":round(config.FT['ThresholdPD'],2),
                     "nTrackedBaselines":nTrackedBaselines,"nInjectedTels":nInjectedTels,
                     "DIT":dit,
                     "minRmsPd":minRmsPd,"maxRmsPd":maxRmsPd,"medianRmsPd":medianRmsPd,
                     "medianRmsDisp":medianRmsDisp,"medianT0":medianT0,
                     "medianFjpBases":medianFjpBases,
                     "minFC":minFC,"maxFC":maxFC,"medianFC":medianFC,
                     "minGamma":minGamma,"maxGamma":maxGamma,"medianGamma":medianGamma,
                     "minFlux":minFlux,"maxFlux":maxFlux,"medianFluxTels":medianFluxTels}
            
            if args.print:
                print("DIT of which te performance are computed:",dit,"seconds.")
                print(nTrackedBaselines,"tracked baselines:",trackedBaselines)
                print("Non-tracked baselines:",InterfArray.BaseNames[trackedBaselinesBoolean==0])
                print("Min rms(PD):",minRmsPd,"µm")
                print("Max rms(PD):",maxRmsPd,"µm")
                print("Median rms(PD):",medianRmsPd,"µm")
                print("Median t0:",medianT0,"ms")
                print("Median FJP:",medianFjpBases,"s")
                print("Minimal contrast loss in visible (100ms):",(1-maxFC)*100,"%")
                print("Worst contrast loss in visible (100ms):",(1-minFC)*100,"%")
                print("Median contrast loss in visible (100ms):",(1-medianFC)*100,"%")
                
                print("\n",nInjectedTels,"injected telescopes:",injectedTels)
                print("Lowest flux:",minFlux)
                print("Highest flux:",maxFlux)
                print("Median flux:",medianFlux)
                
                print("\n",df_tels)
                print("\n",df_bases[trackedBaselinesBoolean])
            
            if not os.path.isdir(dir0+"/perf/"):
                os.makedirs(dir0+"/perf/")
            identifier = filepath.split("SPICAFT.")[-1].split(".fits")[0]
            perfFileName = dir0+"/perf/SPICAFT."+identifier +"_perf.fits"
            
            with fits.open(filepath) as hduL:
                fileHdr = hduL[0].header
                
            writeInFits(perfFileName, fileHdr, hdr_dico, df_bases, df_tels)

