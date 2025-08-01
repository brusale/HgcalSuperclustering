#!/bin/bash

# Simple bash script that does cmsenv then runs the command given in argument

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$CMSSW_BASE"
cmsenv
cd - 

exec "$@"
