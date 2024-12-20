# Full training
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs
mamba activate ticlRegression-gpu

python3 -m dnn_training.trainer -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v15-sampleDump/superclsDumper_\*.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/full-v2 -D cuda:1 -e 200 -b 512 --trainingLossType=binary
python3 -m dnn_training.trainer -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v15-sampleDump/superclsDumper_\*.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/full-v2 -D cuda:1 -e 200 -b 512 --trainingLossType=continuousAssociationScore

#testing single sample
cd /grid_mnt/vol_home/llr/cms/cuisset/hgcal/supercls/repoForJobs
mamba activate ticlRegression-gpu

python3 -m dnn_training.trainer -i /grid_mnt/data_cms_upgrade/cuisset/supercls/alessandro_electrons/supercls-v13/superclsDumper_1.root -o /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/test -D cuda:1 -e 2



mamba activate ticlRegression-gpu
tensorboard --logdir /grid_mnt/data_cms_upgrade/cuisset/supercls/dnn_training/full-v2