import sys
import argparse
import importlib
from pathlib import Path
import FWCore.ParameterSet.Config as cms

parser = argparse.ArgumentParser(
    prog="CMS job starter",
    usage="cmsRun [cmsRun options] cmsJobStarter.py -- --script=... --input=... (note the double-dash)",
    description="Simple cmsRun python config file that loads a given configuration, and updates input and output file paths. "
)

parser.add_argument("--script", "-s", required=True,
    help="Path to the cmsRun script")
parser.add_argument("--input", "-i", default=[], action="append",
    help="The input file for process.source. Can be specified multiple times")
parser.add_argument("--skipEvents", "-S", default=None, type=int,
    help="Skip first n events of file (param to PoolSource)")
parser.add_argument("--maxEvents", "-M", default=None, type=int,
    help="Number of events to process (process.maxEvents.input)")
parser.add_argument("--output-fevt", "-o",
    help="Output path for FEVTDEBUG(HLT) output")
parser.add_argument("--output-dqm", "-q",
    help="Output path for DQMOutput")
parser.add_argument("--output-fileService", "-F",
    help="Output path for TFileService")

parser.add_argument("py_expression", nargs='*',
    help="Additional python statements to be exec'd")

# remove cmsRun arguments
argv = sys.argv[:]
firstArg = -1
for i, arg in enumerate(argv):
    if arg.endswith('.py'):
        firstArg = i+1
    
    if arg == "--":
        firstArg = i+1
        break

print(argv[firstArg:], file=sys.stderr)
args = parser.parse_args(argv[firstArg:])

pathToBaseConfig = Path(args.script)
sys.path.append(str(pathToBaseConfig.resolve().parent))
print(f"Loading configuration from {str(pathToBaseConfig)}", flush=True)
config_module = importlib.import_module(pathToBaseConfig.with_suffix("").name)
process = config_module.process


process.source.fileNames = cms.untracked.vstring(*('file:'+f for f in args.input))

print("Configuration loaded with parameters :")
print("Input files : " + str(args.input))

if args.skipEvents is not None:
    process.source.skipEvents = cms.untracked.uint32(args.skipEvents)
if args.maxEvents is not None:
    try:
        process.maxEvents.input = cms.untracked.int32(args.maxEvents)
    except:
        process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(args.maxEvents))

if args.output_fevt:
    Path(args.output_fevt).resolve().parent.mkdir(parents=True, exist_ok=True)
    try:
        process.FEVTDEBUGHLToutput.fileName = cms.untracked.string('file:'+args.output_fevt)
    except:
        try:
            process.FEVTDEBUGoutput.fileName = cms.untracked.string('file:'+args.output_fevt)
        except:
            raise RuntimeError("Could not set output file name")
    print("Event dump : " + args.output_fevt)

if args.output_dqm:
    Path(args.output_dqm).resolve().parent.mkdir(parents=True, exist_ok=True)
    process.DQMoutput.fileName = cms.untracked.string('file:'+args.output_dqm)
    print("DQM output : " + args.output_dqm)


if args.output_fileService:
    Path(args.output_fileService).resolve().parent.mkdir(parents=True, exist_ok=True)
    process.TFileService.fileName = cms.string('file:'+args.output_fileService)
    print("TFileService output : " + args.output_fileService)


for expr in args.py_expression:
    print("Exec'ing expression : " + expr, flush=True)
    exec(expr)
