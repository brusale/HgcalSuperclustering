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

from analyzer.dumperReader.reader import *
from analyzer.driver.fileTools import *
from analyzer.driver.computations import *
from analyzer.computations.tracksters import tracksters_seedProperties, CPtoTrackster_properties, CPtoTracksterMerged_properties, CPtoTracksterAllShared_properties, TrackstertoCP_properties
from analyzer.energy_resolution.fit import *
import os
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

fileV5 =  "data/CloseByPion0PU/dataC3D/"
fileV4 =  "data/CloseByPion0PU/dataCA/"
fileV3 = "data/CloseByPion0PU/dataFJ/"

OutputDir = "CloseByPion0PU/"
create_directory(OutputDir)

dumperInputV5 = DumperInputManager([
    fileV5
    ], 
    limitFileCount=None,
    )

dumperInputV4 = DumperInputManager([
    fileV4
    ], 
    limitFileCount=None,
    )

dumperInputV3 = DumperInputManager([
    fileV3
    ],
    limitFileCount=None,
    )

resV5 = runComputations([CPtoTrackster_properties, CPtoTracksterAllShared_properties], dumperInputV5, max_workers=1)
resV4 = runComputations([CPtoTrackster_properties, CPtoTracksterAllShared_properties], dumperInputV4, max_workers=1)
resV3 = runComputations([CPtoTrackster_properties, CPtoTracksterAllShared_properties], dumperInputV3, max_workers=1)

recoToSimV5 = runComputations([TrackstertoCP_properties], dumperInputV5, max_workers=1)
recoToSimV3 = runComputations([TrackstertoCP_properties], dumperInputV3, max_workers=1)

mergedV5 = resV5[0]
mergedV4 = resV4[0]
mergedV3 = resV3[0]

mergedR2SV5 = recoToSimV5[0]
mergedR2SV3 = recoToSimV5[0]
print(mergedR2SV5)

fig = plt.figure(figsize = (15,10))
energyBins = 100
etaBins = 50
phiBins = 50


#### BestRECO Plots #####
outputDirTracksterMerged = create_directory(OutputDir + "/trackstersC3D/")
fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy, range = (0,250),  bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.raw_energy, range = (0,250),  bins = energyBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(mergedV3.raw_energy, range = (0,250),  bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')

plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(mergedV3.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(mergedV3.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("phi")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoPhi.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy / mergedV5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.raw_energy / mergedV4.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(mergedV3.raw_energy / mergedV3.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Response w.r.t Regressed")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoEnergyResponse.png")
plt.close()

### Efficient Reco Trackster Plots
# Filter the data based on the condition raw_energy/regressed_energy_CP >= 0.5
filtered_data_V5 = mergedV5[mergedV5['raw_energy'] / mergedV5['regressed_energy_CP'] >= 0.5]
filtered_data_V4 = mergedV4[mergedV4['raw_energy'] / mergedV4['regressed_energy_CP'] >= 0.5]
filtered_data_V3 = mergedV3[mergedV3['raw_energy'] / mergedV3['regressed_energy_CP'] >= 0.5]

fig = plt.figure(figsize = (15,10))
plt.hist(filtered_data_V5.raw_energy, range = (0,250),  bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(filtered_data_V4.raw_energy, range = (0,250),  bins = energyBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(filtered_data_V3.raw_energy, range = (0,250),  bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoRawEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(filtered_data_V5.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(filtered_data_V4.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(filtered_data_V3.barycenter_eta, range = (-1.5, 1.5), bins = etaBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(filtered_data_V5.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(filtered_data_V4.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(filtered_data_V3.barycenter_phi, range = (-np.pi, np.pi),  bins = phiBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("phi")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoPhi.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(filtered_data_V5.raw_energy / filtered_data_V5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(filtered_data_V4.raw_energy / filtered_data_V4.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(filtered_data_V3.raw_energy / filtered_data_V3.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Response w.r.t Regressed")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoEnergyResponse.png")
plt.close()


##### SimTrackster Plots

outputDirSimTracksters = create_directory(OutputDir + "/simTracksters/")
fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy_CP, range = (0,250),  bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.raw_energy_CP, range = (0,250),  bins = energyBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(mergedV3.raw_energy_CP, range = (0,250),  bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimRawEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.regressed_energy_CP, range = (0,250),  bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.regressed_energy_CP, range = (0,250),  bins = energyBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(mergedV3.regressed_energy_CP, range = (0,250),  bins = energyBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')

plt.legend()
plt.ylabel("Entries")
plt.xlabel("Regressed Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimRegressedEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.barycenter_eta_CP, range = (-1.5, 1.5), bins = etaBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.barycenter_eta_CP, range = (-1.5, 1.5), bins = etaBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(mergedV3.barycenter_eta_CP, range = (-1.5, 1.5), bins = etaBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.barycenter_phi_CP, range = (-np.pi, np.pi),  bins = phiBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.barycenter_phi_CP, range = (-np.pi, np.pi),  bins = phiBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
plt.hist(mergedV3.barycenter_phi_CP, range = (-np.pi, np.pi),  bins = phiBins, label = "FastJet", histtype = "step", lw = 2, color = 'green')
plt.legend()
plt.ylabel("Entries")
plt.xlabel("phi")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimPhi.png")
plt.close()

fig = plt.figure(figsize = (15,10))
plt.hist(mergedV5.raw_energy_CP / mergedV5.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = "CLUE3D", histtype = "step", lw = 2, color = 'red')
plt.hist(mergedV4.raw_energy_CP / mergedV4.regressed_energy_CP, range = (0, 1.5),  bins = energyBins, label = "CA", histtype = "step", lw = 2, color = 'blue')
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

bins = [0, 20, 50, 100, 200 ,400]
data = [mergedV5, mergedV4, mergedV3]
lab = ['CLUE3D', 'CA', 'FastJet']

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
filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= 0.5]
filtered_data_V3 = mergedV3[mergedV3['sharedE'] / mergedV3['raw_energy_CP'] >= 0.5]
energyBins = np.linspace(0,400, 22)
energyBins = np.append(energyBins, 600)
plot_ratio_single(filtered_data_V5.raw_energy_CP, mergedV5.raw_energy_CP, energyBins, rangeX = (0,250), label1="CLUE3D", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3D.png")
plot_ratio_single(filtered_data_V4.raw_energy_CP, mergedV4.raw_energy_CP, energyBins, rangeX = (0,250), label1="CA", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCA.png")
plot_ratio_single(filtered_data_V3.raw_energy_CP, mergedV3.raw_energy_CP, energyBins, rangeX = (0,250), label1="FastJet", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFJ.png")

nums = [filtered_data_V5.raw_energy_CP, filtered_data_V4.raw_energy_CP, filtered_data_V3.raw_energy_CP]
dens = [mergedV5.raw_energy_CP,  mergedV4.raw_energy_CP, mergedV3.raw_energy_CP]
plot_ratio_multiple(nums, dens, energyBins, rangeX = (0,250), labels = ["CLUE3D", "CA", "FastJet"], colors = ['blue', 'red', 'green'], xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DCAFJ.png")

# Now we do the same thing for barycenter_eta and barycenter_phi
etaBins = 20
plot_ratio_single(filtered_data_V5.barycenter_eta_CP, mergedV5.barycenter_eta_CP, etaBins, rangeX = (-1.5,1.5), label1="CLUE3D", xlabel="Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DEta.png")
plot_ratio_single(filtered_data_V4.barycenter_eta_CP, mergedV4.barycenter_eta_CP, etaBins, rangeX = (-1.5,1.5), label1="CA", xlabel = "Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCAEta.png")
plot_ratio_single(filtered_data_V3.barycenter_eta_CP, mergedV3.barycenter_eta_CP, etaBins, rangeX = (-1.5,1.5), label1="FastJet", xlabel = "Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFJEta.png")

nums = [filtered_data_V5.barycenter_eta_CP, filtered_data_V4.barycenter_eta_CP, filtered_data_V3.barycenter_eta_CP]
dens = [mergedV5.barycenter_eta_CP,  mergedV4.barycenter_eta_CP, mergedV3.barycenter_eta_CP]
plot_ratio_multiple(nums, dens, etaBins, rangeX = (-1.5,1.5), labels = ["CLUE3D", "CA", "FastJet"], colors = ['blue', 'red', 'green'], xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DCAFJEta.png")

phiBins = 20
plot_ratio_single(filtered_data_V5.barycenter_phi_CP, mergedV5.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="CLUE3D", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DPhi.png")
plot_ratio_single(filtered_data_V4.barycenter_phi_CP, mergedV4.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="CA", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCAPhi.png")
plot_ratio_single(filtered_data_V3.barycenter_phi_CP, mergedV3.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="FastJet", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFJPhi.png")

nums = [filtered_data_V5.barycenter_phi_CP, filtered_data_V4.barycenter_phi_CP, filtered_data_V3.barycenter_phi_CP]
dens = [mergedV5.barycenter_phi_CP,  mergedV4.barycenter_phi_CP, mergedV3.barycenter_phi_CP]
plot_ratio_multiple(nums, dens, phiBins, rangeX = (-np.pi, np.pi), labels = ["CLUE3D", "CA", "FastJet"], colors = ['blue', 'red', 'green'], xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DCAFJPhi.png")

## Fake rate ##
filtered_R2S_data_V5 = mergedR2SV5[mergedR2SV5['score'] > 0.1]
filtered_R2S_data_V3 = mergedR2SV3[mergedR2SV3['score'] > 0.1]
plot_ratio_single(filtered_R2S_data_V5.raw_energy, mergedR2SV5.raw_energy, energyBins, rangeX = (0,250), xlabel="Raw Energy [GeV]", label1="CLUE3D", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeC3D.png")
plot_ratio_single(filtered_R2S_data_V3.raw_energy, mergedR2SV3.raw_energy, energyBins, rangeX = (0,250), xlabel="Raw Energy [GeV]", label1="FastJet", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeFJ.png")

plot_ratio_single(filtered_R2S_data_V5.barycenter_eta, mergedR2SV5.barycenter_eta, etaBins, rangeX = (-1.5,1.5), xlabel="Eta", label1="CLUE3D", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeC3DEta.png")
plot_ratio_single(filtered_R2S_data_V3.barycenter_eta, mergedR2SV3.barycenter_eta, etaBins, rangeX = (-1.5,1.5), xlabel="Eta", label1="FastJet", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeFJEta.png")

plot_ratio_single(filtered_R2S_data_V5.barycenter_phi, mergedR2SV5.barycenter_phi, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="CLUE3D", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeC3DPhi.png")
plot_ratio_single(filtered_R2S_data_V3.barycenter_phi, mergedR2SV3.barycenter_phi, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="FastJet", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeFJPhi.png")


outputDirEffFakeMergeDup = create_directory(OutputDir + "/trackstersC3D/EffFakeMergeDupVarious/")
effTh = [0.5,0.6,0.7,0.8]
colors = ['blue', 'red', 'green', 'orange', 'purple']
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
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,250), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCLUE3D_energy.png", doRatio = False)

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

#same thing for v4
mergedNum = []
mergedDen = []
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.raw_energy_CP)
    mergedDen.append(mergedV4.raw_energy_CP)
    labels.append(f"CA - {th}")
print(len(mergedNum))
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,250), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCA_energy.png", doRatio = False)


mergedNum = []
mergedDen = []
etaBins = 20
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.barycenter_eta_CP)
    mergedDen.append(mergedV4.barycenter_eta_CP)
    labels.append(f"CA - {th}")
plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (-1.5,1.5), labels = labels, colors = colors, xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCA_eta.png", doRatio = False)


mergedNum = []
mergedDen = []
phiBins = 20
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.barycenter_phi_CP)
    mergedDen.append(mergedV4.barycenter_phi_CP)
    labels.append(f"CA - {th}")
plot_ratio_multiple(mergedNum, mergedDen, phiBins, rangeX = (-np.pi,np.pi), labels = labels, colors = colors, xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCA_phi.png", doRatio = False)

mergedNum = []
mergedDen = []
labels = []
for th in effTh:
    filtered_data_V3 = mergedV3[mergedV3['sharedE'] / mergedV3['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V3.raw_energy_CP)
    mergedDen.append(mergedV3.raw_energy_CP)
    labels.append(f"FastJet - {th}")
print(len(mergedNum))
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,250), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFastJet_energy.png", doRatio = False)


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

mergedV5TMP = resV5[1].reset_index()
mergedV4TMP = resV4[1].reset_index()
mergedV3TMP = resV3[1].reset_index()

mergedV5 = mergedV5TMP.groupby(['eventInternal', 'caloparticle_id']).agg(
    raw_energy=('raw_energy', 'sum'),
    sharedE=('sharedE', 'sum'),
    raw_energy_CP=('raw_energy_CP', lambda x: list(x)[0]),
    regressed_energy_CP=('regressed_energy_CP', lambda x: list(x)[0]),
    barycenter_eta_CP=('barycenter_eta_CP', lambda x: list(x)[0]),
    barycenter_phi_CP=('barycenter_phi_CP', lambda x: list(x)[0])
).reset_index()


mergedV4 = mergedV4TMP.groupby(['eventInternal', 'caloparticle_id']).agg(
    raw_energy=('raw_energy', 'sum'),
    sharedE=('sharedE', 'sum'),
    raw_energy_CP=('raw_energy_CP', lambda x: list(x)[0]),
    regressed_energy_CP=('regressed_energy_CP', lambda x: list(x)[0]),
    barycenter_eta_CP=('barycenter_eta_CP', lambda x: list(x)[0]),
    barycenter_phi_CP=('barycenter_phi_CP', lambda x: list(x)[0])
).reset_index()

mergedV3 = mergedV3TMP.groupby(['eventInternal', 'caloparticle_id']).agg(
    raw_energy=('raw_energy', 'sum'),
    sharedE=('sharedE', 'sum'),
    raw_energy_CP=('raw_energy_CP', lambda x: list(x)[0]),
    regressed_energy_CP=('regressed_energy_CP', lambda x: list(x)[0]),
    barycenter_eta_CP=('barycenter_eta_CP', lambda x: list(x)[0]),
    barycenter_phi_CP=('barycenter_phi_CP', lambda x: list(x)[0])
).reset_index()

outputDirTracksterMerged = create_directory(OutputDir + "/sumTrackstersC3D/Responses/")
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
outputDirEffFakeMergeDup = create_directory(OutputDir + "/sumTrackstersC3D/EffFakeMergeDup/")
filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= 0.5]
filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= 0.5]
filtered_data_V3 = mergedV3[mergedV3['sharedE'] / mergedV3['raw_energy_CP'] >= 0.5]
energyBins = np.linspace(0,400, 22)
energyBins = np.append(energyBins, 600)
plot_ratio_single(filtered_data_V5.raw_energy_CP, mergedV5.raw_energy_CP, energyBins, rangeX = (0,250), label1="CLUE3D", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3D.png")
plot_ratio_single(filtered_data_V4.raw_energy_CP, mergedV4.raw_energy_CP, energyBins, rangeX = (0,250), label1="CA", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCA.png")
plot_ratio_single(filtered_data_V3.raw_energy_CP, mergedV3.raw_energy_CP, energyBins, rangeX = (0,250), label1="FastJet", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFJ.png")

nums = [filtered_data_V5.raw_energy_CP, filtered_data_V4.raw_energy_CP, filtered_data_V3.raw_energy_CP]
dens = [mergedV5.raw_energy_CP,  mergedV4.raw_energy_CP, mergedV3.raw_energy_CP]
plot_ratio_multiple(nums, dens, energyBins, rangeX = (0,250), labels = ["CLUE3D", "CA", "FastJet"], colors = ['blue', 'red', 'green'], xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DCAFJ.png")

# Now we do the same thing for barycenter_eta and barycenter_phi
etaBins = 20
plot_ratio_single(filtered_data_V5.barycenter_eta_CP, mergedV5.barycenter_eta_CP, etaBins, rangeX = (-1.5,1.5), label1="CLUE3D", xlabel="Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DEta.png")
plot_ratio_single(filtered_data_V4.barycenter_eta_CP, mergedV4.barycenter_eta_CP, etaBins, rangeX = (-1.5,1.5), label1="CA", xlabel = "Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCAta.png")
plot_ratio_single(filtered_data_V3.barycenter_eta_CP, mergedV3.barycenter_eta_CP, etaBins, rangeX = (-1.5,1.5), label1="FastJet", xlabel = "Eta", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencFJEta.png")

nums = [filtered_data_V5.barycenter_eta_CP, filtered_data_V4.barycenter_eta_CP, filtered_data_V3.barycenter_eta_CP]
dens = [mergedV5.barycenter_eta_CP,  mergedV4.barycenter_eta_CP, mergedV3.barycenter_eta_CP]
plot_ratio_multiple(nums, dens, etaBins, rangeX = (-1.5, 1.5), labels = ["CLUE3D", "CA", "FastJet"], colors = ['blue', 'red', 'green'], xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DCAFJEta.png")

phiBins = 20
plot_ratio_single(filtered_data_V5.barycenter_phi_CP, mergedV5.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="CLUE3D", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DPhi.png")
plot_ratio_single(filtered_data_V4.barycenter_phi_CP, mergedV4.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="CA", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCAPhi.png")
plot_ratio_single(filtered_data_V3.barycenter_phi_CP, mergedV3.barycenter_phi_CP, phiBins, rangeX = (-np.pi, np.pi), xlabel="Phi", label1="FastJet", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFJPhi.png")

nums = [filtered_data_V5.barycenter_phi_CP, filtered_data_V4.barycenter_phi_CP, filtered_data_V3.barycenter_phi_CP]
dens = [mergedV5.barycenter_phi_CP,  mergedV4.barycenter_phi_CP, mergedV3.barycenter_phi_CP]
plot_ratio_multiple(nums, dens, phiBins, rangeX = (-np.pi, np.pi), labels = ["CLUE3D", "CA", "FastJet"], colors = ['blue', 'red', 'green'], xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyC3DCAFJPhi.png")


outputDirEffFakeMergeDup = create_directory(OutputDir + "/sumTrackstersC3D/EffFakeMergeDupVarious/")
effTh = [0.5,0.6,0.7,0.8]
colors = ['blue', 'red', 'green', 'orange', 'purple']
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
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,250), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCLUE3D_energy.png", doRatio = False)

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

#same thing for v4
mergedNum = []
mergedDen = []
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.raw_energy_CP)
    mergedDen.append(mergedV4.raw_energy_CP)
    labels.append(f"CA - {th}")
print(len(mergedNum))
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,250), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCA_energy.png", doRatio = False)


mergedNum = []
mergedDen = []
etaBins = 20
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.barycenter_eta_CP)
    mergedDen.append(mergedV4.barycenter_eta_CP)
    labels.append(f"CA - {th}")
plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (-1.5,1.5), labels = labels, colors = colors, xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCA_eta.png", doRatio = False)


mergedNum = []
mergedDen = []
phiBins = 20
labels = []
for th in effTh:
    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V4.barycenter_phi_CP)
    mergedDen.append(mergedV4.barycenter_phi_CP)
    labels.append(f"CA - {th}")
plot_ratio_multiple(mergedNum, mergedDen, phiBins, rangeX = (-np.pi,np.pi), labels = labels, colors = colors, xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyCA_phi.png", doRatio = False)

## FastJet
mergedNum = []
mergedDen = []
labels = []
for th in effTh:
    filtered_data_V3 = mergedV3[mergedV3['sharedE'] / mergedV3['raw_energy_CP'] >= th]
    mergedNum.append(filtered_data_V3.raw_energy_CP)
    mergedDen.append(mergedV3.raw_energy_CP)
    labels.append(f"FastJet - {th}")
print(len(mergedNum))
plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,250), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyFastJet_energy.png", doRatio = False)


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

## Fake rate ##
filtered_R2S_data_V5 = mergedR2SV5[mergedR2SV5['score'] > 0.1]
filtered_R2S_data_V3 = mergedR2SV3[mergedR2SV3['score'] > 0.1]
plot_ratio_single(filtered_R2S_data_V5.raw_energy, mergedR2SV5.raw_energy, energyBins, rangeX = (0,250), xlabel="Raw Energy [GeV]", label1="CLUE3D", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeC3D.png")
plot_ratio_single(filtered_R2S_data_V3.raw_energy, mergedR2SV3.raw_energy, energyBins, rangeX = (0,250), xlabel="Raw Energy [GeV]", label1="FastJet", color1="blue", saveFileName=f"{outputDirEffFakeMergeDup}/fakeFJ.png")


