###############################################################################
# Benchmark configuration - runs all extrapolators with 1000 events
# Modified from test_extrapolators.py for benchmarking
###############################################################################
from Configurables import (
    TrackHerabExtrapolator,
    TrackKiselExtrapolator,
    TrackLinearExtrapolator,
    TrackParabolicExtrapolator,
    TrackRungeKuttaExtrapolator,
)
from DDDB.CheckDD4Hep import UseDD4Hep
from PyConf.Algorithms import ExtrapolatorTester
from PyConf.application import ApplicationOptions, configure, configure_input
from PyConf.control_flow import CompositeNode

options = ApplicationOptions(_enabled=False)
options.set_input_and_conds_from_testfiledb("MiniBrunel_2018_MinBias_FTv4_DIGI")
options.evt_max = 1000  # Increased for benchmarking
config = configure_input(options)

if UseDD4Hep:
    dd4hepSvc = config["LHCb::Det::LbDD4hep::DD4hepSvc/LHCb::Det::LbDD4hep::DD4hepSvc"]
    dd4hepSvc.DetectorList = ["/world", "Magnet"]

extrapolators = []
ex = ExtrapolatorTester(name="BenchmarkExtrapolators", Extrapolators=extrapolators)

# Configure all extrapolators for benchmarking
extrapolators += [
    TrackRungeKuttaExtrapolator("Reference"),
    TrackRungeKuttaExtrapolator("BogackiShampine3", RKScheme="BogackiShampine3"),
    TrackRungeKuttaExtrapolator("Verner7", RKScheme="Verner7"),
    TrackRungeKuttaExtrapolator("Verner9", RKScheme="Verner9"),
    TrackRungeKuttaExtrapolator("Tsitouras5", RKScheme="Tsitouras5"),
    TrackKiselExtrapolator("Kisel"),
    TrackHerabExtrapolator("Herab"),
    TrackLinearExtrapolator("Linear"),
    TrackParabolicExtrapolator("Parabolic"),
]

config.update(configure(options, CompositeNode("TopSeq", [ex])))

print("="*80)
print("BENCHMARK CONFIGURATION")
print("="*80)
print(f"Events: {options.evt_max}")
print(f"Extrapolators: {len(extrapolators)}")
for e in extrapolators:
    print(f"  - {e.name()}")
print("="*80)
