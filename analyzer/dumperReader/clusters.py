import awkward as ak
import pandas as pd
import numpy as np
from typing import Union

from .assocs import assocs_toDf

clusters_basic_fields = ["seedID", "energy", "position_x", "position_y", "position_z", "position_eta", "position_phi", "cluster_layer_id", "cluster_type", ]#"cluster_time", "cluster_number_of_hits"]

def clusters_toDf(clusters:ak.Array) -> pd.DataFrame:
    """ Makes a dataframe with all clusters
    Index : eventInternal, cluster_id
    """
    assert clusters.ndim <= 2, "Clusters ak.Array should not include hits information (or other nested fields)"
    try:
        if "cluster_id" in clusters.fields:
           return (ak.to_dataframe(clusters,
               levelname=lambda x : {0:"eventInternal", 1:"cluster_id_wrong"}[x])
               .reset_index("cluster_id_wrong", drop=True)
               .set_index("cluster_id", append=True)
           )
        else:
           return ak.to_dataframe(clusters,
                levelname=lambda x : {0:"eventInternal", 1:"cluster_id"}[x])
    except KeyError as e:
        if e.args[0] == 2:
            raise ValueError("Clusters ak.Array should not include hits information (or other nested fields") from e
        else:
            raise e

def _convertLcToDataframe(clusters:Union[ak.Array,pd.DataFrame]) -> pd.DataFrame:
    """ Convert if needed an ak.Array of clusters to dataframe """
    if isinstance(clusters, ak.Array):
        return clusters_toDf(clusters)
    else:
        return clusters

def clusters_joinWithCaloParticles(clusters:ak.Array, caloParticles:ak.Array, assoc:ak.Array, score_threshold=assocs_toDf.__defaults__[0]) -> pd.DataFrame:
    """ Make a merged dataframe holding trackster information joined with the caloparticles information.
    Only clusters with an association are kept.
    Index : eventInternal cluster_id
    """
    df_merged_1 = pd.merge(
        clusters_toDf(clusters),
        assocs_toDf(assoc, score_threshold=0.),
        left_index=True, right_index=True)
    return pd.merge(
        df_merged_1,
        caloparticle_toDf(caloParticles),
        left_on=["eventInternal", "cp_id"],  right_index=True,
        how="left",
        suffixes=(None, "_sim")
    )
