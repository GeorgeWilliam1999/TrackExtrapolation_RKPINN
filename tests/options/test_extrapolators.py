###############################################################################
# (c) Copyright 2020-2022 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from Configurables import (
    TrackHerabExtrapolator,
    TrackKiselExtrapolator,
    TrackMasterExtrapolator,
    TrackRungeKuttaExtrapolator,
    TrackSimpleExtraSelector,
)
from DDDB.CheckDD4Hep import UseDD4Hep
from PyConf.Algorithms import ExtrapolatorTester
from PyConf.application import ApplicationOptions, configure, configure_input
from PyConf.control_flow import CompositeNode

options = ApplicationOptions(_enabled=False)
options.set_input_and_conds_from_testfiledb("MiniBrunel_2018_MinBias_FTv4_DIGI")
options.evt_max = 1
config = configure_input(options)

if UseDD4Hep:
    dd4hepSvc = config["LHCb::Det::LbDD4hep::DD4hepSvc/LHCb::Det::LbDD4hep::DD4hepSvc"]
    dd4hepSvc.DetectorList = ["/world", "Magnet"]

# if UseDD4Hep:
#     from Configurables import LHCb__Tests__FakeRunNumberProducer as FET
#     from Configurables import LHCb__Det__LbDD4hep__IOVProducer as IOVProducer
#     odin_path = '/Event/DummyODIN'
#     all = [
#         FET('FakeRunNumber', ODIN=odin_path, Start=42, Step=20),
#         IOVProducer("ReserveIOVDD4hep", ODIN=odin_path)
#     ]

extrapolators = []
ex = ExtrapolatorTester(name="ExtrapolatorTester", Extrapolators=extrapolators)
extrapolators += [
    TrackRungeKuttaExtrapolator("Reference"),
    TrackRungeKuttaExtrapolator("BogackiShampine3", RKScheme="BogackiShampine3"),
    TrackRungeKuttaExtrapolator("Verner7", RKScheme="Verner7"),
    TrackRungeKuttaExtrapolator("Verner9", RKScheme="Verner9"),
    TrackRungeKuttaExtrapolator("Tsitouras5", RKScheme="Tsitouras5", OutputLevel=1),
    TrackKiselExtrapolator("Kisel"),
    TrackHerabExtrapolator("Herab"),
    # TrackRKPINNExtrapolator temporarily disabled - needs Gaudi component registration
    # TrackRKPINNExtrapolator("PINN", 
    #                          ModelPath="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/pinn_model_cpp.bin",
    #                          HiddenLayers=[128, 128, 64]),
]

config.update(configure(options, CompositeNode("TopSeq", [ex])))
