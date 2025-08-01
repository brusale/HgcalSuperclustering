import sys
sys.path.append("../..")
from functools import partial
from typing import Literal

import uproot
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import hist
from seaborn import regplot

from analyzer.dumperReader.reader import *
from analyzer.driver.fileTools import *
from analyzer.driver.computations import *
from analyzer.computations.tracksters import CPtoTracksterMerged_properties, MergedTrackstertoCP_properties
from analyzer.computations.clusters import CPtoLayerCluster_properties, LayerClustertoCP_properties, CPtoLayerClusterAllShared_properties
from analyzer.energy_resolution.fit import *
import os
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib import cm
from utilities import *

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return directory_path

#fileV5 = "data/CloseByPhoton200PU_PF/dataC3D/"
#fileV3 = "data/CloseByPhoton200PU_PF/dataC3D/"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cache", action="store_true", help="Use cached dataframes")
args = parser.parse_args()

#fileV5 = "data150X/CloseByPhoton200PU/dataC3D/"
#fileV3 = "data150X/CloseByPhoton200PU/histoFJ/"
#fileC2D = "data150X/CloseByPhoton200PU/dataC3D/"
#filePF = "data150X/CloseByPhoton200PU/dataPF/"

fileV5 = "data151X/CloseByPion0PU/"
fileV3 = "data151X/CloseByPion0PU/"
fileC2D = "data151X/CloseByPion0PU/"
filePF = "data151X/CloseByPion0PU/"


#filePF = "data/CloseByPhoton200PU_PF/dataPF/"
#fileC2D = "data/CloseByPhoton200PU_PF/dataC2D/"

OutputDir = "CloseByPion0PU_151X_linking/"
create_directory(OutputDir)
if not args.cache:
    dumperInputV5 = DumperInputManager([
        fileV5
        ], 
        limitFileCount=5,
        )

    dumperInputV3 = DumperInputManager([
        fileV3
        ],
        limitFileCount=5,
        )
    
    dumperInputPF = DumperInputManager([
        filePF
        ],
        limitFileCount=5,
        )
    
    dumperInputLC = DumperInputManager([
        fileC2D,
        ],
        limitFileCount=5,
        )
    
    resV5 = runComputations([CPtoTracksterMerged_properties], dumperInputV5, max_workers=10)
    resV3 = runComputations([CPtoTracksterMerged_properties], dumperInputV3, max_workers=10)
    
    recoToSimV5 = runComputations([MergedTrackstertoCP_properties], dumperInputV5, max_workers=10)
    recoToSimV3 = runComputations([MergedTrackstertoCP_properties], dumperInputV3, max_workers=10)
    
    
    
    mergedV5 = resV5[0]
    #mergedV5 = mergedV5[mergedV5['score'] <= 0.1]
    mergedV3 = resV3[0]
    #mergedV3 = mergedV3[mergedV3['score'] <= 0.1]
    
    mergedR2SV5 = recoToSimV5[0]
    mergedR2SV3 = recoToSimV3[0]
    #print(list(mergedV5.columns.values))
    
    if not os.path.exists('cache/'):
        os.mkdir('cache/')
    mergedV5.to_pickle('cache/CloseByPhoton200PU_CLUE3D_simToReco.pkl')
    mergedV3.to_pickle('cache/CloseByPhoton200PU_FastJet_simToReco.pkl')
    mergedR2SV5.to_pickle('cache/CloseByPhoton200PU_CLUE3D_recoToSim.pkl')
    mergedR2SV3.to_pickle('cache/CloseByPhoton200PU_FastJet_recoToSim.pkl')
else:
    print(" ")
    mergedV5 = pd.read_pickle('cache/CloseByPhoton200PU_CLUE3D_simToReco.pkl') 
    mergedV3 = pd.read_pickle('cache/CloseByPhoton200PU_FastJet_simToReco.pkl') 
    mergedR2SV5 = pd.read_pickle('cache/CloseByPhoton200PU_CLUE3D_recoToSim.pkl')
    mergedR2SV3 = pd.read_pickle('cache/CloseByPhoton200PU_FastJet_recoToSim.pkl')
    mergedLC = pd.read_pickle('cache/CloseByPhoton200PU_CLUE2D_simToReco.pkl')
    mergedPF = pd.read_pickle('cache/CloseByPhoton200PU_PF_simToReco.pkl')


#mergedV5 = mergedV5[mergedV5['score'] <= 0.1]
#mergedV3 = mergedV3[mergedV3['score'] <= 0.1]

fig = plt.figure(figsize = (15,10))
energyBins = 200
etaBins = 50
phiBins = 50

energy_edges = [[0, 10], [10, 25], [25, 50], [50, 100], [100, 200], [200, 500]]


#### BestRECO Plots #####
outputDirTracksterMerged = create_directory(OutputDir + "/trackstersC3D/")
fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy, range=(0,1000),  bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV3.raw_energy, range=(0,1000),  bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergy.png")
plt.close()

fig = plt.figure(figsize=(15,10))
plt.hist(mergedV5.raw_em_energy, range=(0,1000), bins=energyBins, label="CLUE3D", histtype="step", lw=2, color='red')
plt.hist(mergedV3.raw_em_energy, range=(0,1000), bins=energyBins, label="FastJet", histtype="step", lw=2, color='green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw EM Energy [GeV]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoRawEMEnergy.png")
plt.close()

fig = plt.figure(figsize=(15,10))
plt.hist(mergedV5.raw_em_energy/mergedV5.raw_energy, bins=energyBins, label="CLUE3D", histtype="step", lw=2, color='red')
plt.hist(mergedV3.raw_em_energy/mergedV3.raw_energy, bins=energyBins, label="FastJet", histtype="step", lw=2, color='green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("h/e")
plt.yscale('log')
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoRAWEMFraction.png")
plt.close()

'''
fig = plt.figure(figsize=(15,10))
plt.hist2d(mergedV5.raw_energy, mergedV5.raw_em_energy/mergedV5.raw_energy,bins=energyBins, norm=colors.LogNorm())
plt.colorbar()
plt.legend()
plt.ylabel("EM raw energy fraction")
plt.xlabel("Raw Energy [GeV]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoRAWEMFractionVsEC3D.png")
plt.close()

fig = plt.figure(figsize=(15,10))
plt.hist2d(mergedV3.raw_energy, mergedV3.raw_em_energy/mergedV3.raw_energy,bins=energyBins, norm=colors.LogNorm())
plt.colorbar()
plt.legend()
plt.ylabel("EM raw energy fraction")
plt.xlabel("Raw Energy [GeV]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoRAWEMFractionVsEFJ.png")
plt.close()
'''
fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV3.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV3.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("phi")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoPhi.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy / mergedV5.regressed_energy_CP, range = (0, 5), bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV3.raw_energy / mergedV3.regressed_energy_CP, range = (0, 5), bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Response w.r.t Regressed")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoEnergyResponse.png")
plt.close()

fig = plt.figure(figsize=(15,10))
plt.hist(mergedV5.e25_over_e55, range=(0,1), bins=energyBins, label="CLUE3D", histtype="step", lw=2, color='red')
plt.hist(mergedV3.e25_over_e55, range=(0,1), bins=energyBins, label="FastJet", histtype="step", lw=2, color='green')
#plt.hist(mergedV5.e25_over_e55_CP, range=(0,1), bins=energyBins, label="CLUE3D - SimTrackstersCP", histtype="step", lw=2, color='red', linestyle="--")
plt.legend()
plt.ylabel("Entries")
plt.xlabel(r"E$_{2x5}$/E$_{5x5}$")
hep.cms.text("Simulation", loc=0)
plt.yscale('log')
plt.savefig(outputDirTracksterMerged + "E25OverE55.png")
plt.close()


fig = plt.figure(figsize= (15,10))
plt.hist(mergedV5.n_vertices, range=(0, 50), bins = 50, label = "CLUE3D", histtype = "step", lw = 2, color = 'red', density=True)
plt.hist(mergedV5.n_vertices_CP, range=(0, 50), bins = 50, label = "CLUE3D - SimTrackstersCP", histtype = "step", lw = 2, color = 'red', linestyle="--", density=True)
plt.hist(mergedV3.n_vertices, range=(0, 50), bins = 50, label = "FastJet", histtype = "step", lw = 2, color = 'green', density=True)
plt.hist(mergedV3.n_vertices_CP, range=(0, 50), bins = 50, label = "FastJet - SimTrackstersCP", histtype = "step", lw = 2, color = 'green', linestyle="--", density=True)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("NClusters / Trackster")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "NClustersPerTrackster.png")
plt.close()

fig = plt.figure(figsize= (15,10))
plt.hist(mergedV5.span, range=(0, 6), bins = 6, label = "CLUE3D", histtype = "step", lw = 2, color = 'red', density=True)
plt.hist(mergedV5.span_CP, range=(0, 6), bins = 6, label = "CLUE3D - SimTrackstersCP", histtype = "step", lw = 2, color = 'red', linestyle="--", density=True)
plt.hist(mergedV3.span, range=(0, 6), bins = 6, label = "FastJet", histtype = "step", lw = 2, color = 'green', density=True)
plt.hist(mergedV3.span_CP, range=(0, 6), bins = 6, label = "FastJet - SimTrackstersCP", histtype = "step", lw = 2, color = 'green', linestyle="--", density=True)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("N unique layers")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "NUniqueLayersPerTrackster.png")
plt.close()

fig = plt.figure(figsize= (15,10))
plt.hist(mergedV5.min_layer, range=(0, 6), bins = 6, label = "CLUE3D", histtype = "step", lw = 2, color = 'red', density=True)
plt.hist(mergedV5.min_layer_CP, range=(0, 6), bins = 6, label = "CLUE3D - SimTrackstersCP", histtype = "step", lw = 2, color = 'red', linestyle="--", density=True)
plt.hist(mergedV3.min_layer, range=(0, 6), bins = 6, label = "FastJet", histtype = "step", lw = 2, color = 'green', density=True)
plt.hist(mergedV3.min_layer_CP, range=(0, 6), bins = 6, label = "FastJet - SimTrackstersCP", histtype = "step", lw = 2, color = 'green', linestyle="--", density=True)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("First layer")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "FirstLayerPerTrackster.png")
plt.close()

fig = plt.figure(figsize= (15,10))
plt.hist(mergedV5.max_layer, range=(0, 6), bins = 6, label = "CLUE3D", histtype = "step", lw = 2, color = 'red', density=True)
plt.hist(mergedV5.max_layer_CP, range=(0, 6), bins = 6, label = "CLUE3D - SimTrackstersCP", histtype = "step", lw = 2, color = 'red', linestyle="--", density=True)
plt.hist(mergedV3.max_layer, range=(0, 6), bins = 6, label = "FastJet", histtype = "step", lw = 2, color = 'green', density=True)
plt.hist(mergedV3.max_layer_CP, range=(0, 6), bins = 6, label = "FastJet - SimTrackstersCP", histtype = "step", lw = 2, color = 'green', linestyle="--", density=True)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Last layer")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "LastLayerPerTrackster.png")
plt.close()

fig = plt.figure(figsize=(15,10))
barycenter_bins = np.linspace(120, 200, 40)
plt.hist(np.sqrt(mergedV5.barycenter_x**2+mergedV5.barycenter_y**2), bins=barycenter_bins, label="CLUE3D", histtype="step", lw=2, color="red", density=True)
plt.hist(np.sqrt(mergedV5.barycenter_x_CP**2+mergedV5.barycenter_y_CP**2), bins=barycenter_bins, label="CLUE3D - SimTrackstersCP", histtype="step", lw=2, color="red", linestyle="--", density=True)
plt.hist(np.sqrt(mergedV3.barycenter_x**2+mergedV3.barycenter_y**2), bins=barycenter_bins, label="FastJet", histtype="step", lw=2, color="green", density=True)
plt.hist(np.sqrt(mergedV3.barycenter_x_CP**2+mergedV3.barycenter_y_CP**2), bins=barycenter_bins, label="FastJet - SimTrackstersCP", histtype="step", lw=2, color="green", linestyle="--", density=True)
plt.legend()
plt.axvline(x=180.6, c="black", linestyle="--", label="HCAL")
plt.yscale('log')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Barycenter (xy plane)")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BarycenterXYPerTrackster.png")
plt.close()

def BarycenterVSEnergy(energy, barycenter, name, title):
    fig = plt.figure(figsize=(15,10))
    plt.hist2d(energy, barycenter, bins=(energyBins, 20),norm=colors.LogNorm(), label=title)
    plt.colorbar()
    plt.ylim(100,300) 
    plt.xlim(0, 250)
    plt.axhline(y=180.6, c="black", linestyle="--", label="HCAL")
    plt.legend()
    plt.xlabel("Raw energy [GeV]")
    plt.ylabel("Barycenter [cm]")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + name)

#BarycenterVSEnergy(mergedV5.raw_energy, np.sqrt(mergedV5.barycenter_x**2+mergedV5.barycenter_y**2), "BarycenterXYEnergyPerTracksterC3D.png", "CLUE3D")
#BarycenterVSEnergy(mergedV5.raw_energy_CP, np.sqrt(mergedV5.barycenter_x_CP**2+mergedV5.barycenter_y_CP**2), "BarycenterXYEnergyPerSimTracksterC3D.png", "CLUE3D - SimTrackstersCP")
#BarycenterVSEnergy(mergedV3.raw_energy, np.sqrt(mergedV3.barycenter_x**2+mergedV3.barycenter_y**2), "BarycenterXYEnergyPerTracksterFJ.png", "FastJet")
#BarycenterVSEnergy(mergedV3.raw_energy_CP, np.sqrt(mergedV3.barycenter_x_CP**2+mergedV3.barycenter_y_CP**2), "BarycenterXYEnergyPerSimTracksterFJ.png", "FastJet - SimTrackstersCP")

for edges in energy_edges:
    energy_binnedV5 = mergedV5[(mergedV5['regressed_energy_CP'] > edges[0]) & (mergedV5['regressed_energy_CP'] <= edges[1])]
    energy_binnedV3 = mergedV3[(mergedV3['regressed_energy_CP'] > edges[0]) & (mergedV3['regressed_energy_CP'] <= edges[1])]
    fig = plt.figure(figsize = (15,10))
    plt.hist(energy_binnedV5.raw_energy, range=(0,1000),  bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
    plt.hist(energy_binnedV3.raw_energy, range=(0,1000),  bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Raw Energy [Gev]")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergy{}To{}.png".format(edges[0], edges[1]))
    plt.close()

    fig = plt.figure(figsize = (15,10))
    plt.hist(energy_binnedV5.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
    plt.hist(energy_binnedV3.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("eta")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoEta{}To{}.png".format(edges[0], edges[1]))
    plt.close()

    fig = plt.figure(figsize = (15,10))
    plt.hist(energy_binnedV5.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
    plt.hist(energy_binnedV3.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("phi")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoPhi{}To{}.png".format(edges[0], edges[1]))
    plt.close()

    fig = plt.figure(figsize = (15,10))
    plt.hist(energy_binnedV5.raw_energy / energy_binnedV5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
    plt.hist(energy_binnedV3.raw_energy / energy_binnedV3.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Response w.r.t Regressed")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoEnergyResponse{}To{}.png".format(edges[0], edges[1]))
    plt.close()

### Efficient Reco Trackster Plots
# Filter the data based on the condition raw_energy/regressed_energy_CP >= 0.5
filtered_data_V5 = mergedV5[mergedV5['raw_energy'] / mergedV5['regressed_energy_CP'] >= 0.5]
filtered_data_V3 = mergedV3[mergedV3['raw_energy'] / mergedV3['regressed_energy_CP'] >= 0.5]

fig = plt.figure(figsize = (15,10))
plt.hist(filtered_data_V5.raw_energy, range=(0,1000),  bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(filtered_data_V3.raw_energy, range=(0,1000),  bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoRawEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(filtered_data_V5.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(filtered_data_V3.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(filtered_data_V5.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(filtered_data_V3.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("phi")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoPhi.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(filtered_data_V5.raw_energy / filtered_data_V5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(filtered_data_V3.raw_energy / filtered_data_V3.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Response w.r.t Regressed")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoEnergyResponse.png")
plt.close()

for edges in energy_edges:
    energy_binnedV5 = mergedV5[(mergedV5['regressed_energy_CP'] > edges[0]) & (mergedV5['regressed_energy_CP'] <= edges[1])]
    energy_binnedV3 = mergedV3[(mergedV3['regressed_energy_CP'] > edges[0]) & (mergedV3['regressed_energy_CP'] <= edges[1])]

    fig = plt.figure(figsize = (15,10))
    plt.hist(energy_binnedV5.raw_energy, range=(0,1000),  bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
    plt.hist(energy_binnedV3.raw_energy, range=(0,1000),  bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Raw Energy [Gev]")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "EfficientRecoRawEnergy{}To{}.png".format(edges[0], edges[1]))
    plt.close()

    fig = plt.figure(figsize = (15,10))
    plt.hist(energy_binnedV5.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
    plt.hist(energy_binnedV3.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("eta")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "EfficientRecoEta{}To{}.png".format(edges[0], edges[1]))
    plt.close()

    fig = plt.figure(figsize = (15,10))
    plt.hist(energy_binnedV5.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
    plt.hist(energy_binnedV3.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("phi")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "EfficientRecoPhi{}To{}.png".format(edges[0], edges[1]))
    plt.close()

    fig = plt.figure(figsize = (15,10))
    plt.hist(energy_binnedV5.raw_energy / energy_binnedV5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
    plt.hist(energy_binnedV3.raw_energy / energy_binnedV3.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Response w.r.t Regressed")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "EfficientRecoEnergyResponse{}To{}.png".format(edges[0], edges[1]))
    plt.close()

##### SimTrackster Plots

outputDirSimTracksters = create_directory(OutputDir + "/simTracksters/")
fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy_CP, range=(0,1000),  bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV3.raw_energy_CP, range=(0,1000),  bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimRawEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.regressed_energy_CP, range=(0,1000),  bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV3.regressed_energy_CP, range=(0,1000),  bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')

plt.legend()
plt.ylabel("Entries")
plt.xlabel("Regressed Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimRegressedEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.barycenter_eta_CP, range = (-1.5, 1.5), bins = etaBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV3.barycenter_eta_CP, range = (-1.5, 1.5), bins = etaBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.barycenter_phi_CP, range = (-np.pi, np.pi),  bins = phiBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV3.barycenter_phi_CP, range = (-np.pi, np.pi),  bins = phiBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("phi")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimPhi.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy_CP / mergedV5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV3.raw_energy_CP / mergedV3.regressed_energy_CP, range = (0, 1.5),  bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Response w.r.t Regressed")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimEnergyResponse.png")
plt.close()

#### Responses ####
# Get the "viridis" colormap
import matplotlib as mpl

cmap = mpl.colormaps['Set1']

# Create a custom colormap with the desired number of colors
custom_cmap = ListedColormap(cmap(np.linspace(0, 1, 10)))

# Set the color cycle
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=custom_cmap.colors)

bins = [0, 20, 50, 100, 200]
data = [mergedV5, mergedV3]
lab = ['CLUE3D', 'FastJet']
outputDirTracksterMerged = create_directory(OutputDir + "/trackstersC3D/Responses/")
for id, d in enumerate(data):
    fig = plt.figure(figsize = (15,10))
    for b in bins:
        filtered_data_V5 = d[d['regressed_energy_CP'] >= b]
        plt.hist(filtered_data_V5.raw_energy / filtered_data_V5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "SimTrackster >= " + str(b) + " GeV", histtype = "step", lw = 2, density = True)
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Raw Response w.r.t Regressed")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergyResponse" + lab[id] + ".png")
    plt.close()

for id, d in enumerate(data):
    fig = plt.figure(figsize = (15,10))
    for b in bins:
        filtered_data_V5 = d[d['regressed_energy_CP'] >= b]
        plt.hist(filtered_data_V5.raw_energy / filtered_data_V5.raw_energy_CP, range = (0, 1.5), bins = energyBins, label = "SimTrackster >= " + str(b) + " GeV", histtype = "step", lw = 2, density = True)
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Raw Response w.r.t Raw")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergyResponseWrtRaw" + lab[id] + ".png")
    plt.close()


## Efficiencies ##
outputDirEffFakeMergeDup = create_directory(OutputDir + "/trackstersC3D/EffFakeMergeDup/")
filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= 0.5]
filtered_data_V3 = mergedV3[mergedV3['sharedE'] / mergedV3['raw_energy_CP'] >= 0.5]
energyBins = np.linspace(0,250, 22)
#energyBins = np.append(energyBins, 600)
plot_ratio_single(filtered_data_V5.raw_energy_CP, mergedV5.raw_energy_CP, energyBins, rangeX = (0,1000), label1="CLUE3D", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3D.png")
plot_ratio_single(filtered_data_V3.raw_energy_CP, mergedV3.raw_energy_CP, energyBins, rangeX = (0,1000), label1="FastJet", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFJ.png")

nums = [filtered_data_V5.raw_energy_CP,filtered_data_V3.raw_energy_CP]
dens = [mergedV5.raw_energy_CP, mergedV3.raw_energy_CP]
plot_ratio_multiple(nums, dens, energyBins, rangeX = (0,1000), labels = ["CLUE3D", "FastJet"], colors = ['blue', 'red', 'green'], xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DFJ.png")

# Now we do the same thing for barycenter_eta and barycenter_phi
etaBins = 20
plot_ratio_single(filtered_data_V5.barycenter_eta_CP, mergedV5.barycenter_eta_CP, etaBins, rangeX = (-1.5,1.5), label1="CLUE3D", xlabel="Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DEta.png")
plot_ratio_single(filtered_data_V3.barycenter_eta_CP, mergedV3.barycenter_eta_CP, etaBins, rangeX = (-1.5,1.5), label1="FastJet", xlabel = "Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFJEta.png")

nums = [filtered_data_V5.barycenter_eta_CP, filtered_data_V3.barycenter_eta_CP]
dens = [mergedV5.barycenter_eta_CP, mergedV3.barycenter_eta_CP]
plot_ratio_multiple(nums, dens, etaBins, rangeX = (-1.5,1.5), labels = ["CLUE3D", "FastJet"], colors = ['blue', 'red', 'green'], xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DFJEta.png")

phiBins = 20
plot_ratio_single(filtered_data_V5.barycenter_phi_CP, mergedV5.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="CLUE3D", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DPhi.png")
plot_ratio_single(filtered_data_V3.barycenter_phi_CP, mergedV3.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="FastJet", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFJPhi.png")

nums = [filtered_data_V5.barycenter_phi_CP, filtered_data_V3.barycenter_phi_CP]
dens = [mergedV5.barycenter_phi_CP, mergedV3.barycenter_phi_CP]
plot_ratio_multiple(nums, dens, phiBins, rangeX = (-np.pi, np.pi), labels = ["CLUE3D", "FastJet"], colors = ['blue', 'red', 'green'], xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DFJPhi.png")

Rbins = 50
nums = [np.sqrt(filtered_data_V5.barycenter_x_CP**2+filtered_data_V5.barycenter_y_CP**2), np.sqrt(filtered_data_V3.barycenter_x_CP**2+filtered_data_V3.barycenter_y_CP**2)]
dens = [np.sqrt(mergedV5.barycenter_x_CP**2+mergedV5.barycenter_y_CP**2), np.sqrt(mergedV3.barycenter_x_CP**2+mergedV3.barycenter_y_CP**2)]
plot_ratio_multiple(nums, dens, Rbins, rangeX=(128, 300), labels=["CLUE3D", "FastJet"], colors=['blue', 'red', 'green'], xlabel="Barycenter R [cm]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DFJR.png")
## Fake rate ##
filtered_R2S_data_V5 = mergedR2SV5[mergedR2SV5['score'] > 0.6]
filtered_R2S_data_V3 = mergedR2SV3[mergedR2SV3['score'] > 0.6]
plot_ratio_single(filtered_R2S_data_V5.raw_energy, mergedR2SV5.raw_energy, energyBins, rangeX = (0,1000), xlabel="Raw Energy [GeV]", label1="CLUE3D", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeC3D.png")
plot_ratio_single(filtered_R2S_data_V3.raw_energy, mergedR2SV3.raw_energy, energyBins, rangeX = (0,1000), xlabel="Raw Energy [GeV]", label1="FastJet", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeFJ.png")

plot_ratio_single(filtered_R2S_data_V5.barycenter_eta, mergedR2SV5.barycenter_eta, etaBins, rangeX = (-1.5,1.5), xlabel="Eta", label1="CLUE3D", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeC3DEta.png")
plot_ratio_single(filtered_R2S_data_V3.barycenter_eta, mergedR2SV3.barycenter_eta, etaBins, rangeX = (-1.5,1.5), xlabel="Eta", label1="FastJet", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeFJEta.png")

plot_ratio_single(filtered_R2S_data_V5.barycenter_phi, mergedR2SV5.barycenter_phi, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="CLUE3D", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeC3DPhi.png")
plot_ratio_single(filtered_R2S_data_V3.barycenter_phi, mergedR2SV3.barycenter_phi, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="FastJet", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeFJPhi.png")

nums = [filtered_R2S_data_V5.raw_energy, filtered_R2S_data_V3.raw_energy]
dens = [mergedR2SV5.raw_energy, mergedR2SV3.raw_energy]
plot_ratio_multiple(nums, dens, energyBins, rangeX = (0, 250), labels = ["CLUE3D", "FastJet"], colors = ['blue', 'red', 'green'], xlabel="Raw Energy [GeV]", ylabel="Fake rate", saveFileName=f"{outputDirEffFakeMergeDup}/fakeC3DFJ.png")

outputDirEffFakeMergeDup = create_directory(OutputDir + "/trackstersC3D/EffFakeMergeDupVarious/")
effTh = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
colors = ['pink', 'cyan', 'olive', 'magenta', 'blue', 'red', 'green', 'orange', 'purple']
mergedNum = []
mergedDen = []
energyBins = np.linspace(0,400, 22)
energyBins = np.append(energyBins, 600)
labels = []
for th in effTh:
    filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V5.raw_energy_CP)
    mergedDen.append(mergedV5.raw_energy_CP)
    labels.append(f"CLUE3D - {th}")
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,1000), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCLUE3D_energy.png", doRatio = False)

mergedNum = []
mergedDen = []
etaBins = 20
labels = []
for th in effTh:
    filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V5.barycenter_eta_CP)
    mergedDen.append(mergedV5.barycenter_eta_CP)
    labels.append(f"CLUE3D - {th}")
plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (-1.5,1.5), labels = labels, colors = colors, xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCLUE3D_eta.png", doRatio = False)


mergedNum = []
mergedDen = []
phiBins = 20
labels = []
for th in effTh:
    filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V5.barycenter_phi_CP)
    mergedDen.append(mergedV5.barycenter_phi_CP)
    labels.append(f"CLUE3D - {th}")
plot_ratio_multiple(mergedNum, mergedDen, phiBins, rangeX = (-np.pi,np.pi), labels = labels, colors = colors, xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCLUE3D_phi.png", doRatio = False)

#same thing for v3
mergedNum = []
mergedDen = []
labels = []
for th in effTh:
    filtered_data_V3 = mergedV3[mergedV3['sharedE'] / mergedV3['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V3.raw_energy_CP)
    mergedDen.append(mergedV3.raw_energy_CP)
    labels.append(f"FastJet - {th}")
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,1000), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFastJet_energy.png", doRatio = False)


mergedNum = []
mergedDen = []
etaBins = 20
labels = []
for th in effTh:
    filtered_data_V3 = mergedV3[mergedV3['sharedE'] / mergedV3['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V3.barycenter_eta_CP)
    mergedDen.append(mergedV3.barycenter_eta_CP)
    labels.append(f"FastJet - {th}")
plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (-1.5,1.5), labels = labels, colors = colors, xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFastJet_eta.png", doRatio = False)


mergedNum = []
mergedDen = []
phiBins = 20
labels = []
for th in effTh:
    filtered_data_V3 = mergedV3[mergedV3['sharedE'] / mergedV3['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V3.barycenter_phi_CP)
    mergedDen.append(mergedV3.barycenter_phi_CP)
    labels.append(f"FastJet - {th}")
plot_ratio_multiple(mergedNum, mergedDen, phiBins, rangeX = (-np.pi,np.pi), labels = labels, colors = colors, xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFastJet_phi.png", doRatio = False)

### Trackster sums

bins = [0, 20, 50, 100, 200, 500, 1000]


## Fake rate ##
filtered_R2S_data_V5 = mergedR2SV5[mergedR2SV5['score'] > 0.1]
filtered_R2S_data_V3 = mergedR2SV3[mergedR2SV3['score'] > 0.1]
plot_ratio_single(filtered_R2S_data_V5.raw_energy, mergedR2SV5.raw_energy, energyBins, rangeX = (0,1000), xlabel="Raw Energy [GeV]", label1="CLUE3D", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeC3D.png")
plot_ratio_single(filtered_R2S_data_V3.raw_energy, mergedR2SV3.raw_energy, energyBins, rangeX = (0,1000), xlabel="Raw Energy [GeV]", label1="FastJet", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeFJ.png")

nums = [filtered_R2S_data_V5.raw_energy, filtered_R2S_data_V3.raw_energy]
dens = [mergedR2SV5.raw_energy, mergedR2SV3.raw_energy]
plot_ratio_multiple(nums, dens, energyBins, rangeX = (0, 1000), labels = ["CLUE3D", "FastJet"], colors = ['blue', 'red', 'green'], xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/fakeC3DFJ.png")


filtered_R2S_data_V5 = mergedR2SV5[mergedR2SV5['score'] <= 0.1]
mergedR2SV5.to_pickle("dump.pkl")
