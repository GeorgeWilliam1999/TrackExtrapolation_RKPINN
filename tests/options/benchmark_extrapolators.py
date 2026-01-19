###############################################################################
# Benchmark configuration for track extrapolators
# Based on test_extrapolators.py - runs comprehensive timing and accuracy tests
###############################################################################
from Configurables import (
    TrackHerabExtrapolator,
    TrackKiselExtrapolator,
    TrackLinearExtrapolator,
    TrackMasterExtrapolator,
    TrackParabolicExtrapolator,
    TrackRungeKuttaExtrapolator,
    TrackSimpleExtraSelector,
)
from DDDB.CheckDD4Hep import UseDD4Hep
from PyConf.Algorithms import ExtrapolatorTester
from PyConf.application import ApplicationOptions, configure, configure_input
from PyConf.control_flow import CompositeNode

# Use testfiledb for proper conditions/detector setup (same as working test)
options = ApplicationOptions(_enabled=False)
options.set_input_and_conds_from_testfiledb("MiniBrunel_2018_MinBias_FTv4_DIGI")
options.evt_max = 1  # Only need 1 event - ExtrapolatorTester generates test tracks internally
config = configure_input(options)

# Configure detector - only need magnet for extrapolation
if UseDD4Hep:
    dd4hepSvc = config["LHCb::Det::LbDD4hep::DD4hepSvc/LHCb::Det::LbDD4hep::DD4hepSvc"]
    dd4hepSvc.DetectorList = ["/world", "Magnet"]

# Configure all extrapolators to benchmark
extrapolators = []
ex = ExtrapolatorTester(name="ExtrapolatorTester", Extrapolators=extrapolators)

# Add extrapolators with timing enabled
extrapolators += [
    # Reference methods (high accuracy)
    TrackRungeKuttaExtrapolator("Reference"),  # Default RK4
    
    # Alternative RK schemes
    TrackRungeKuttaExtrapolator("BogackiShampine3", RKScheme="BogackiShampine3"),
    TrackRungeKuttaExtrapolator("Verner7", RKScheme="Verner7"),
    TrackRungeKuttaExtrapolator("Verner9", RKScheme="Verner9"),
    TrackRungeKuttaExtrapolator("Tsitouras5", RKScheme="Tsitouras5", OutputLevel=1),
    
    # Fast analytic approximations
    TrackKiselExtrapolator("Kisel"),
    TrackHerabExtrapolator("Herab"),
    TrackLinearExtrapolator("Linear"),
    TrackParabolicExtrapolator("Parabolic"),
]

# Configure the control flow
config.update(configure(options, CompositeNode("TopSeq", [ex])))
