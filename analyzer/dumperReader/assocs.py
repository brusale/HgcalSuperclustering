from functools import cached_property
from typing import Literal

import awkward as ak
import pandas as pd



def assocs_zip_recoToSim(assocs_unzipped:ak.Array, simVariant:Literal["CP", "SC"]="CP") -> ak.Array:
    """ Zip associations array into records

    Parameters : 
     - assocs_unzipped : the input awkward array
     - simVariant : can be "CP" (CaloParticle, the default) in which case use associations to SimTrackster from CaloParticles.
                    can be "SC" (SimCluster) in which case use assocs to SimCluster

    From type: nevts * {
        tsCLUE3DEM_recoToSim_SC: var * var * uint32,
        tsCLUE3DEM_recoToSim_SC_score: var * var * float32,
        tsCLUE3DEM_recoToSim_SC_sharedE: var * var * float32
    }
    To type: nevts * var (nbOfTsForCurEvent) * var (nbOfAssocsForCurTs) * {
        ts_id: int64,
        caloparticle_id/simcluster_id: uint32,
        score: float32,
        sharedE: float32
    }
    """
    if simVariant == "CP":
        return ak.zip({
            "ts_id":ak.local_index(assocs_unzipped.tsCLUE3D_recoToSim_CP, axis=1),
            "caloparticle_id":assocs_unzipped.tsCLUE3D_recoToSim_CP,
            "score":assocs_unzipped.tsCLUE3D_recoToSim_CP_score,
            "sharedE":assocs_unzipped.tsCLUE3D_recoToSim_CP_sharedE})  
    elif simVariant == "SC":
        return ak.zip({
            "ts_id":ak.local_index(assocs_unzipped.tsCLUE3D_recoToSim_SC, axis=1),
            "simcluster_id":assocs_unzipped.tsCLUE3D_recoToSim_SC,
            "score":assocs_unzipped.tsCLUE3D_recoToSim_SC_score,
            "sharedE":assocs_unzipped.tsCLUE3D_recoToSim_SC_sharedE})
    else:
        raise ValueError("CP or SC")
    
def assocs_zip_simToReco(assocs_unzipped:ak.Array, simVariant:Literal["CP", "SC"]="CP") -> ak.Array:
    """ Zip associations array into records
    
    Parameters : 
     - assocs_unzipped : the input awkward array
     - simVariant : can be "CP" (CaloParticle, the default) in which case use associations to SimTrackster from CaloParticles.
                    can be "SC" (SimCluster) in which case use assocs to SimCluster
    From type: nevts * {
        tsCLUE3DEM_simToReco_CP: var * var * uint32,
        tsCLUE3DEM_simToReco_CP_score: var * var * float32,
        tsCLUE3DEM_simToReco_CP_sharedE: var * var * float32
    }
    To type: nevts * var (nbOfCPPerEvent = 2 normally for CP) * var (nbOfAssocsForCurCP) * {
        caloparticle_id/simcluster_id: int64,
        ts_id: uint32,
        score: float32,
        sharedE: float32
    }
    """
    if simVariant == "CP":
        return ak.zip({
            "caloparticle_id":ak.local_index(assocs_unzipped.tsCLUE3D_simToReco_CP, axis=1),
            "ts_id":assocs_unzipped.tsCLUE3D_simToReco_CP,
            "score":assocs_unzipped.tsCLUE3D_simToReco_CP_score,
            "sharedE":assocs_unzipped.tsCLUE3D_simToReco_CP_sharedE})
    elif simVariant == "SC":
        return ak.zip({
            "simcluster_id":ak.local_index(assocs_unzipped.tsCLUE3D_simToReco_SC, axis=1),
            "ts_id":assocs_unzipped.tsCLUE3D_simToReco_SC,
            "score":assocs_unzipped.tsCLUE3D_simToReco_SC_score,
            "sharedE":assocs_unzipped.tsCLUE3D_simToReco_SC_sharedE})
    else:
        raise ValueError("CP or SC")
def assocs_zip_recoMergedToSim(assocs_unzipped:ak.Array, simVariant:Literal["CP", "SC"]="CP") -> ak.Array:
    """ Zip associations array into records

    Parameters : 
     - assocs_unzipped : the input awkward array
     - simVariant : can be "CP" (CaloParticle, the default) in which case use associations to SimTrackster from CaloParticles.
                    can be "SC" (SimCluster) in which case use assocs to SimCluster

    From type: nevts * {
        tsCLUE3DEM_recoToSim_SC: var * var * uint32,
        tsCLUE3DEM_recoToSim_SC_score: var * var * float32,
        tsCLUE3DEM_recoToSim_SC_sharedE: var * var * float32
    }
    To type: nevts * var (nbOfTsForCurEvent) * var (nbOfAssocsForCurTs) * {
        ts_id: int64,
        caloparticle_id/simcluster_id: uint32,
        score: float32,
        sharedE: float32
    }
    """
    if simVariant == "CP":
        return ak.zip({
            "ts_id":ak.local_index(assocs_unzipped.Mergetracksters_recoToSim_CP, axis=1),
            "caloparticle_id":assocs_unzipped.Mergetracksters_recoToSim_CP,
            "score":assocs_unzipped.Mergetracksters_recoToSim_CP_score,
            "sharedE":assocs_unzipped.Mergetracksters_recoToSim_CP_sharedE})  
    elif simVariant == "SC":
        return ak.zip({
            "ts_id":ak.local_index(assocs_unzipped.Mergetstracksters_recoToSim_SC, axis=1),
            "simcluster_id":assocs_unzipped.Mergetstracksters_recoToSim_SC,
            "score":assocs_unzipped.Mergetstracksters_recoToSim_SC_score,
            "sharedE":assocs_unzipped.Mergetstracksters_recoToSim_SC_sharedE})
    else:
        raise ValueError("CP or SC")
    
def assocs_zip_simToRecoMerged(assocs_unzipped:ak.Array, simVariant:Literal["CP", "SC"]="CP") -> ak.Array:
    """ Zip associations array into records
    
    Parameters : 
     - assocs_unzipped : the input awkward array
     - simVariant : can be "CP" (CaloParticle, the default) in which case use associations to SimTrackster from CaloParticles.
                    can be "SC" (SimCluster) in which case use assocs to SimCluster
    From type: nevts * {
        tsCLUE3DEM_simToReco_CP: var * var * uint32,
        tsCLUE3DEM_simToReco_CP_score: var * var * float32,
        tsCLUE3DEM_simToReco_CP_sharedE: var * var * float32
    }
    To type: nevts * var (nbOfCPPerEvent = 2 normally for CP) * var (nbOfAssocsForCurCP) * {
        caloparticle_id/simcluster_id: int64,
        ts_id: uint32,
        score: float32,
        sharedE: float32
    }
    """
    if simVariant == "CP":
        return ak.zip({
            "caloparticle_id":ak.local_index(assocs_unzipped.Mergetracksters_simToReco_CP, axis=1),
            "ts_id":assocs_unzipped.Mergetracksters_simToReco_CP,
            "score":assocs_unzipped.Mergetracksters_simToReco_CP_score,
            "sharedE":assocs_unzipped.Mergetracksters_simToReco_CP_sharedE})
    elif simVariant == "SC":
        return ak.zip({
            "simcluster_id":ak.local_index(assocs_unzipped.Mergetstracksters_simToReco_SC, axis=1),
            "ts_id":assocs_unzipped.Mergetstracksters_simToReco_SC,
            "score":assocs_unzipped.Mergetstracksters_simToReco_SC_score,
            "sharedE":assocs_unzipped.Mergetstracksters_simToReco_SC_sharedE})
    else:
        raise ValueError("CP or SC")
def assocs_dropOnes(assocs_zipped:ak.Array) -> ak.Array:
    """ Drops associations of score one (worst score)
    
    Type : type: nevts * var (nbOfCP/TSPerEvent) * var (nbOfAssocsForCurCP/TS) * {
        caloparticle_id/simcluster_id: int64,
        ts_id: uint32,
        score: float32,
        sharedE: float32
    }
    """
    ar = assocs_zipped[assocs_zipped.score < 1]
    # Drop empty lists (Tracksters/CP that have no assocation with score < 1)
    return ar[ak.num(ar, axis=-1) > 0]

def assocs_bestScore(assocs_zipped:ak.Array) -> ak.Array:
    """ Selects the association with the best (lowest) score for each trackster (or for each SimTrackster) 

    Trackster (resp SimTrackster) with no associations are dropped (therefore ts_id can be different from the index in the list)
    From type: nevts * var (nbOf(Sim)TsForCurEvent) * var (nbOfAssocsForCur(Sim)Ts) * {
        ts_id: int64,
        caloparticle_id/simcluster_id: uint32,
        score: float32,
        sharedE: float32
    }
    to type: nevts * var (nbOf(Sim)TsForCurEvent) * {
        ts_id: int64,
        caloparticle_id/simcluster_id: uint32,
        score: float32,
        sharedE: float32
    }
    """
    idx_sort = ak.argsort(assocs_zipped.score, ascending=True) # sort by score
    # Take the first assoc, putting None in case a trackster has no assocation
    # Then drop the None
    return ak.drop_none(ak.firsts(assocs_zipped[idx_sort], axis=-1), axis=-1)
    

def assocs_toDf(assocs_zipped:ak.Array, score_threshold=0.5) -> pd.DataFrame:
    """ Makes a pandas dataframe of associations

    Parameters : 
     - score_threshold : can be a float in [0, 1], in which case it will keep only associations with score lower than this threshold (lower score = better).
                        If None, will not place any cut on score
    Index : eventInternal, ts_id
    Columns: caloparticle_id/simcluster_id score sharedE
    """
    assocs = assocs_bestScore(assocs_zipped)
    if score_threshold is not None:
        assocs = assocs[assocs_bestScore(assocs_zipped).score < score_threshold]
    return (ak.to_dataframe(
        assocs,
        levelname=lambda x : {0:"eventInternal", 1:"mapping"}[x])
        .reset_index("mapping", drop=True).set_index("ts_id", append=True)
    )