from analyzer.driver.computations import DataframeComputation
from analyzer.dumperReader.reader import DumperReader, clusters_basic_fields, CPToLayerClusterProperties, LayerClusterToCPProperties
from analyzer.dumperReader.clusters import clusters_toDf
from analyzer.driver.fileTools import SingleInputReader

#cannot use a lambda as multiprocessing does nor work due to pickle issues
def _CPtoLayerCluster_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return CPToLayerClusterProperties(reader.assocs_bestScore_simToRecoLayerClusters_df, reader.clusters_zipped[clusters_basic_fields], reader.caloparticles_df)
CPtoLayerCluster_properties = DataframeComputation(_CPtoLayerCluster_fct, "CPtoLayerCluster_properties")

def _LayerClusterToCP_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return LayerClusterToCPProperties(reader.assocs_bestScore_recoToSimLayerClusters_df(dropOnes=False), reader.clusters_zipped[clusters_basic_fields], reader.caloparticles_df)
LayerClustertoCP_properties = DataframeComputation(_LayerClusterToCP_fct, "LayerClusterToCP_properties")

def _CPtoLayerClusterAllShared_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return CPToLayerClusterProperties(reader.assocs_bestScore_simToRecoSharedLayerClusters_df, reader.clusters_zipped[clusters_basic_fields], reader.caloparticles_df)
CPtoLayerClusterAllShared_properties = DataframeComputation(_CPtoLayerClusterAllShared_fct, "CPtoLayerClusterAllShared_properties")
