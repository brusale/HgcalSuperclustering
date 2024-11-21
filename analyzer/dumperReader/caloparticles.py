import awkward as ak
import pandas as pd
import numpy as np
from typing import Union

from .assocs import assocs_toDf

caloparticle_basic_fields = ["caloparticle_energy", "caloparticle_et", "caloparticle_pt", "caloparticle_eta", "caloparticle_phi"]

def caloparticles_df(caloparticles:ak.Array) -> pd.DataFrame:
    """ Makes a dataframe with all caloparticles
    Index : eventInternal, ts_id
    """
    assert caloparticles.ndim == 2,
    try:
        if "caloparticle_id" in caloparticles.fields:
           return (ak.to_dataframe(caloparticles,
               levelname=lambda x : {0:"eventInternal", 1:"cp_id_wrong"}[x])
               .reset_index("caloparticle_id_wrong", drop=True)
               .set_index("caloparticle_id", append=True)
           )
        else:
            return ak.to_dataframe(caloparticles,
                levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id"}[x])
    except KeyError as e:
        if e.args[0] == 2:
            raise ValueError("CaloParticles ak.Array should not include hits information (or other nested fields)") from e
       else:
           raise e


def _convertCpToDataframe(caloparticles:Union[ak.Array,pd.DataFrame]) -> pd.DataFrame:
    """ Convert if needed an ak.Array of caloparticles to dataframe """
    if isinstance(caloparticles, ak.Array):
        return caloparticles_toDf(caloparticles)
    else:
        return caloparticles

