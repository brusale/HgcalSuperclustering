import awkward as ak
import pandas as pd
from typing import Union

from .tracksters import supercluster_joinTracksters, _convertTsToDataframe
from .clusters import _convertLcToDataframe
from .assocs import assocs_bestScore, assocs_zip_recoToSim


def superclusterToSim_df(supercluster_df:pd.DataFrame, assocs_bestScore_recoToSim_df:pd.DataFrame, tracksters_df:pd.DataFrame) -> pd.DataFrame:
    """ Make Df of tracksters in superclusters joined with recoToSim associations and trackster information
    
    Index : eventInternal	supercls_id	ts_in_supercls_id	
    Columns : ts_id	simts_id	score	sharedE	raw_energy	regressed_energy"""
    df = supercluster_df.join(assocs_bestScore_recoToSim_df(), on=["eventInternal", "ts_id"])
    df.score = df.score.fillna(1)
    df.sharedE = df.sharedE.fillna(0)
    
    return supercluster_joinTracksters(df, tracksters_df)

def TracksterToCPProperties(assocs_bestScore_recoToSim_df:pd.DataFrame, tracksters:Union[ak.Array, pd.DataFrame], simTrackstersCP_df:pd.DataFrame):
    """ For each CaloParticle, get the best associated trackster properties
    Parameters :
     - tracksters : dataframe (or zipped akward array) of tracksters with properties to keep
    """
    return (assocs_bestScore_recoToSim_df.join(_convertTsToDataframe(tracksters), on=["eventInternal", "ts_id"]).join(simTrackstersCP_df, rsuffix="_CP"))

def LayerClusterToCPProperties(assocs_bestScore_recoToSim_df:pd.DataFrame, clusters:Union[ak.Array, pd.DataFrame], caloparticles_df:pd.DataFrame):
    return (assocs_bestScore_recoToSim_df.join(_convertLcToDataframe(clusters), on=["eventInternal", "cluster_id"]).join(clustersCP_df))
