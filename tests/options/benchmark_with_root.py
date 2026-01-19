###############################################################################
# Production benchmark configuration for track extrapolators
# Writes ROOT ntuples for detailed analysis and publication plots
###############################################################################
from Configurables import (
    ApplicationMgr,
    NTupleSvc,
    HistogramPersistencySvc,
    TrackHerabExtrapolator,
    TrackKiselExtrapolator,
    TrackLinearExtrapolator,
    TrackParabolicExtrapolator,
    TrackRungeKuttaExtrapolator,
    TrackExtrapolatorTesterSOA,
)
from DDDB.CheckDD4Hep import UseDD4Hep
from PyConf.application import ApplicationOptions, configure_input

# Use testfiledb for proper conditions/detector setup
options = ApplicationOptions(_enabled=False)
options.set_input_and_conds_from_testfiledb("MiniBrunel_2018_MinBias_FTv4_DIGI")
options.evt_max = 1  # ExtrapolatorTesterSOA generates test tracks internally
config = configure_input(options)

# Configure detector - only need magnet for extrapolation
if UseDD4Hep:
    dd4hepSvc = config["LHCb::Det::LbDD4hep::DD4hepSvc/LHCb::Det::LbDD4hep::DD4hepSvc"]
    dd4hepSvc.DetectorList = ["/world", "Magnet"]

# List of extrapolators to benchmark
extrapolator_names = [
    "TrackRungeKuttaExtrapolator/Reference",           # Default RK4 - baseline
    "TrackRungeKuttaExtrapolator/BogackiShampine3",    # RK3 scheme
    "TrackRungeKuttaExtrapolator/Verner7",             # 7th order RK
    "TrackRungeKuttaExtrapolator/Verner9",             # 9th order RK  
    "TrackRungeKuttaExtrapolator/Tsitouras5",          # 5th order adaptive RK
    "TrackKiselExtrapolator/Kisel",                    # Fast analytic
    "TrackHerabExtrapolator/Herab",                    # Helix approximation
    "TrackLinearExtrapolator/Linear",                  # Straight-line (for comparison)
    "TrackParabolicExtrapolator/Parabolic",            # Parabolic (for comparison)
]

# Configure the individual extrapolators
TrackRungeKuttaExtrapolator("BogackiShampine3", RKScheme="BogackiShampine3")
TrackRungeKuttaExtrapolator("Verner7", RKScheme="Verner7")
TrackRungeKuttaExtrapolator("Verner9", RKScheme="Verner9")
TrackRungeKuttaExtrapolator("Tsitouras5", RKScheme="Tsitouras5")
TrackKiselExtrapolator("Kisel")
TrackHerabExtrapolator("Herab")
TrackLinearExtrapolator("Linear")
TrackParabolicExtrapolator("Parabolic")

# Configure benchmark algorithm with ROOT output
benchmark = TrackExtrapolatorTesterSOA(
    "BenchmarkExtrapolators",
    ReferenceExtrapolator="TrackRungeKuttaExtrapolator/Reference",  # Compare against RK4
    Extrapolators=extrapolator_names,
    InitialZ=3000.0,  # mm - typical LHCb detector start
    FinalZ=7000.0,    # mm - 4m propagation distance
)

# Configure application with ROOT histogram output
ApplicationMgr(
    TopAlg=[benchmark],
    EvtSel="NONE",
    EvtMax=1,
    HistogramPersistency="ROOT",
)

# Configure NTuple output service
NTupleSvc(
    Output=[
        "FILE1 DATAFILE='benchmark_results.root' OPT='NEW' TYP='ROOT'"
    ]
)

print("=" * 80)
print("Benchmark Configuration")
print("=" * 80)
print(f"Output file: benchmark_results.root")
print(f"Test tracks: 121 (11x11 grid)")
print(f"Propagation: z = 3000 → 7000 mm (4m)")
print(f"Extrapolators: {len(extrapolator_names)}")
print("=" * 80)
