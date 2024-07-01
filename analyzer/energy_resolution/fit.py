from functools import partial
import dataclasses
from dataclasses import dataclass

from scipy.optimize import curve_fit
import numpy as np
import hist
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

@dataclass
class CruijffParam:
    A:float
    """ Amplitude"""
    m:float
    """ Central value """
    sigmaL:float
    """ Left tail sigma """
    sigmaR:float
    """ Right tail sigma """
    alphaL:float
    """ Left tail alpha """
    alphaR:float
#     sigmaEff:float
    

    @property
    def sigmaAverage(self) -> float:
        return (self.sigmaL + self.sigmaR) / 2
    
    def makeTuple(self) -> tuple[float]:
        return dataclasses.astuple(self)

@dataclass
class CruijffFitResult:
    params:CruijffParam
    covMatrix:np.ndarray

def cruijff(x, A, m, sigmaL,sigmaR, alphaL, alphaR):
    dx = (x-m)
    SL = np.full(x.shape, sigmaL)
    SR = np.full(x.shape, sigmaR)
    AL = np.full(x.shape, alphaL)
    AR = np.full(x.shape, alphaR)
    sigma = np.where(dx<0, SL,SR)
    alpha = np.where(dx<0, AL,AR)
    if(sigmaL == sigmaR):
        sigma = sigmaL
    f = 2*sigma*sigma + alpha*dx*dx
    return A* np.exp(-dx*dx/f)

# def fitCruijff(h_forFit:hist.Hist, sigmaEff:bool=False) -> CruijffFitResult:
#     mean = np.average(h_forFit.axes[0].centers, weights=h_forFit.values())
#     stdDev = np.average((h_forFit.axes[0].centers - mean)**2, weights=h_forFit.values())
#     param_optimised,param_covariance_matrix = curve_fit(cruijff, h_forFit.axes[0].centers, h_forFit.values(), 
#         p0=[np.max(h_forFit), mean, stdDev, stdDev,  0.1, 0.05], sigma=np.maximum(np.sqrt(h_forFit.values()), 1.8), absolute_sigma=True, maxfev=500000,
#         #bounds=np.transpose([(0., np.inf), (-np.inf, np.inf), (0., np.inf), (0., np.inf), (-np.inf, np.inf), (-np.inf, np.inf)])
#         )
#     # Calculate initial sigma_eff
#     if(sigmaEff):
#         sigmaL, sigmaR = param_optimised[2], param_optimised[3]
#         sigma_eff = (sigmaL + sigmaR) / 2
#         interval_width = 0.683 * np.sum(h_forFit.values())  # 68.3% of the measurement

#         # Iteratively adjust sigmaL and sigmaR until approximately 68.3% of the measurement lies within the interval
#         while True:
#             # Calculate the range for 68.3% of the measurement
#             lower_bound = mean - sigma_eff / 2
#             upper_bound = mean + sigma_eff / 2

#             # Calculate the sum of entries above and below the interval
#             above_interval_sum = np.sum(h_forFit.values()[h_forFit.axes[0].centers > upper_bound])
#             below_interval_sum = np.sum(h_forFit.values()[h_forFit.axes[0].centers < lower_bound])

#             # Calculate the total sum within the interval
#             fraction_within_interval = (np.sum(h_forFit.values()) - above_interval_sum - below_interval_sum)

            
#             # Check if the fraction within the interval is approximately 68.3%
#             print(lower_bound, upper_bound, fraction_within_interval, interval_width)
#             if fraction_within_interval >= interval_width * 0.99 and fraction_within_interval <= interval_width * 1.01:
#                 break  # Exit loop if the condition is met

#             # Adjust sigmaL and sigmaR
#             if fraction_within_interval < interval_width:
#                 sigmaL *= 1.01
#                 sigmaR *= 1.01
#             else:
#                 sigmaL *= 0.99
#                 sigmaR *= 0.99

#             # Update sigma_eff
#             sigma_eff = (sigmaL + sigmaR) / 2
#             param_optimised = np.append(param_optimised, sigma_eff)
#         # Update the optimized parameters with the adjusted sigmaL and sigmaR
#         print(fraction_within_interval)
#         param_optimised[2], param_optimised[3] = sigmaL, sigmaR

#         # Return the fit result with sigma_eff
#         return CruijffFitResult(CruijffParam(*param_optimised), param_covariance_matrix)
#     else:        
#         return CruijffFitResult(CruijffParam(*param_optimised), param_covariance_matrix)

def fitCruijff(h_forFit: hist.Hist, sigmaEff: bool = False) -> CruijffFitResult:
    mean = np.average(h_forFit.axes[0].centers, weights=h_forFit.values())
    stdDev = np.average((h_forFit.axes[0].centers - mean) ** 2, weights=h_forFit.values())
    param_optimised, param_covariance_matrix = curve_fit(
        cruijff, h_forFit.axes[0].centers, h_forFit.values(),
        p0=[np.max(h_forFit), mean, stdDev, stdDev, 0.1, 0.05],
        sigma=np.maximum(np.sqrt(h_forFit.values()), 1.8),
        absolute_sigma=True, maxfev=500000,
    )
    
    if sigmaEff:
        nbins, binw, xmin = len(h_forFit.axes[0].centers), h_forFit.axes[0].widths[0], h_forFit.axes[0].edges[0]
        mu, rms, total = mean, np.sqrt(stdDev), np.sum(h_forFit.values())

        nWindow = int(rms / binw) if (rms / binw) < 0.1 * nbins else int(0.1 * nbins)
        rlim = 0.683 * total
        wmin, iscanmin = 9999999, -999
        sigmaEff = 0
        for iscan in range(-1 * nWindow, nWindow + 1):
            i_centre = int((mu - xmin) / binw + 1 + iscan)
            x_centre = (i_centre - 0.5) * binw + xmin
            x_up, x_down = x_centre, x_centre
            i_up, i_down = i_centre, i_centre
            y = h_forFit.values()[i_centre]
            r = y
            reachedLimit = False

            for j in range(1, nbins):
                if reachedLimit:
                    continue

                if i_up < nbins and not reachedLimit:
                    i_up += 1
                    x_up += binw
                    y = h_forFit.values()[i_up]
                    r += y
                    if r > rlim:
                        reachedLimit = True
                else:
                    print(" --> Reached nBins in effSigma calc. Returning 0 for effSigma")
                    return (CruijffFitResult(CruijffParam(*param_optimised), param_covariance_matrix), 0)

                if not reachedLimit:
                    if i_down > 0:
                        i_down -= 1
                        x_down -= binw
                        y = h_forFit.values()[i_down]
                        r += y
                        if r > rlim:
                            reachedLimit = True
                    else:
                        print(" --> Reached 0 in effSigma calc. Returning 0 for effSigma")
                        return (CruijffFitResult(CruijffParam(*param_optimised), param_covariance_matrix), 0)
                        

            if y == 0.:
                dx = 0.
            else:
                dx = (r - rlim) * (binw / y)

            w = (x_up - x_down + binw - dx) * 0.5
            if w < wmin:
                wmin = w
        sigmaL, sigmaR = param_optimised[2], param_optimised[3]
        sigma_Cruj = (sigmaL + sigmaR) / 2
        sigma_eff = wmin
#         param_optimised = np.append(param_optimised,sigma_eff)
        print(param_optimised)
    else:
        sigma_eff = None

    return (CruijffFitResult(CruijffParam(*param_optimised), param_covariance_matrix), sigma_eff)


eratio_axis = partial(hist.axis.Regular, 100, 0., 3.5, name="e_ratio")
eta_axis = hist.axis.Variable([0, 3.5], name="absSeedEta", label="|eta|seed")
eratio_eff_axis = partial(hist.axis.Regular, 20, 0., 500, name="e_ratio") 
# seedPt_axis = hist.axis.Variable([ 10.,  20.,  30.,  50.,  70., 100., 150., 200., 250., 300., 350.,
#        400., 450., 500.], name="seedPt", label="Seed Et (GeV)") # edges are computed so that there are the same number of events in each bin
def make_scOrTsOverCP_energy_histogram(name, minEn, maxEn, label=None):
    seedPt_axis = hist.axis.Variable([minEn, maxEn], name="seedPt", label=f"Seed Et {maxEn}(GeV)") # edges are computed so that there are the same number of events in each bin
    h = hist.Hist(eratio_axis(label=label), name=name, label=label)
    return h

def fill_scOverCP_energy_histogram(h:hist.Hist, df:pd.DataFrame, minEn:float, maxEn:float):
    """ df should be CPtoSC_df ie CaloParticle to Supercluster """
    dfN = df.where((df.regressed_energy_CP >= minEn) & (df.regressed_energy_CP <= maxEn)).dropna()
    h.fill(e_ratio=dfN.raw_energy/dfN.regressed_energy_CP)

def fill_seedTsOverCP_energy_histogram(h:hist.Hist, df:pd.DataFrame):
    """ df should be CPtoTs_df ie CaloParticle to seed trackster (highest pt trackster for each endcap) """
    h.fill(e_ratio=df.raw_energy/df.regressed_energy_CP,
        seedPt=df.regressed_energy_CP) 

def make_num_eff_histo(name, bins, minVar, maxVar, label=None):
    eratio_eff_axis = partial(hist.axis.Regular, bins, minVar, maxVar, name="e_ratio")  
    h = hist.Hist(eratio_eff_axis(label=label), name=name, label=label)
    return h

def fill_num_eff_histo(h:hist.Hist, df:pd.DataFrame, minShared:float, var:str, corrected:bool):
    """ df should be CPtoSC_df ie CaloParticle to Supercluster """
    if(corrected == False):
        dfN = df.where((df.sharedE / df.raw_energy_CP >= minShared))
    else:
        dfN = df.where((df.sharedE / df.tot_sharedE >= minShared))
    h.fill(getattr(dfN, var))  
           
def fill_den_eff_histo(h:hist.Hist, df:pd.DataFrame, minShared:float, var:str):
    """ df should be CPtoSC_df ie CaloParticle to Supercluster """
#     print(getattr(df,var))
    h.fill(getattr(df, var))


def fitMultiHistogram(h:list[hist.Hist], sigmaEff:bool=False) -> list[list[CruijffFitResult]]:
    """ Cruijff fit of multi-dimensional histogram of Supercluster/CaloParticle energy """
    res = []
    sigmasEff = []
#     for eta_bin in range(len(h[0].axes["absSeedEta"])):
        
    for i in range(len(h)):
        h_1d = h[i]
        res.append([])
        sigmasEff.append([])
#         for seedPt_bin in range(len(h)):
#             print(eta_bin, seedPt_bin)
#             h_1d = h[seedPt_bin][{"absSeedEta":eta_bin, "seedPt":0}]
        fitR, sEff = fitCruijff(h_1d, sigmaEff)
        res[-1] = fitR
        sigmasEff[-1] = sEff
    return (res, sigmasEff)

def plotSingleHistWithFit(h_1d:hist.Hist, fitRes:CruijffFitResult, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    hep.histplot([h_1d], label=["Best associated trackster"], ax=ax, yerr=False, flow="none")
    x_plotFct = np.linspace(h_1d.axes[0].centers[0], h_1d.axes[0].centers[-1], 500)
    # print(fitRes)
#     sigma = fitRes.params.sigmaEff if fitRes.params.sigmaEff != 0 else fitRes.params.sigmaAverage
    # ax.plot(x_plotFct, cruijff(x_plotFct,*fitRes.params.makeTuple()), label=f"Cruijff fit\n$\sigma={fitRes.params.sigmaAverage:.3f}$")
    # ax.set_xlim(0.1, 3.5)
    # ax.set_ylabel("Events")
    # ax.legend()
    hep.cms.text("Preliminary", exp="TICLv5", ax=ax)
    hep.cms.lumitext("PU=0", ax=ax)

bin_edgesEnergy = [(9.5, 10.5), (19.5, 20.5), (29.5, 30.5), (49.5, 50.5), (69.5, 70.5), (99.5, 100.5), (199.5, 200.5), (399.5, 400.5), (599.5, 600.5)]

def ptBinToText(ptBin:int) -> str:
    low, high = bin_edgesEnergy[ptBin]
    return r"$E_{\text{gen}} \in \left[" + f"{low:.3g}; {high:.3g}" + r"\right]$"

def etaBinToTextEqual(etaBin:float) -> str:
    return r"$|\eta_{\text{gen}}| =" + f"{etaBin}$"
def plotAllFits(h:list[hist.Hist], fitResults:list[CruijffFitResult], etaFloat:float, bin_edgesEnergy, outputDir):

    for i in range(len(h)):
        seedPt_binT = bin_edgesEnergy[i]       
        h_1d = h[i]
        eta_bins =  (1.89,1.91)
        eta_bin = 0
        plotSingleHistWithFit(h_1d, fitResults[0][i])
        plt.text(0.05, 0.95, etaBinToTextEqual(etaFloat)+"\n"+ptBinToText(i), va="top", transform=plt.gca().transAxes, fontsize=20)
        plt.savefig(f"{outputDir}/{seedPt_binT[0]}_{seedPt_binT[1]}_{eta_bin}.png")