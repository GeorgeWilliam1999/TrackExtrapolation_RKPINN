###############################################################################
# Production benchmark: NN field map variants vs trilinear baseline
# Writes ROOT ntuples with per-track timing and sub-operation breakdowns.
#
# Variants:
#   1. Trilinear interpolation (baseline)
#   2. NN scalar SiLU
#   3. NN scalar ReLU
#   4. NN AVX2 ReLU
#
# All use CashKarp RK scheme with EnableSubTimers=True.
# TrackExtrapolatorTesterSOA generates synthetic test states on a grid.
###############################################################################
from DDDB.CheckDD4Hep import UseDD4Hep
from PyConf.Algorithms import TrackExtrapolatorTesterSOA
from PyConf.Tools import TrackRungeKuttaExtrapolator
from PyConf.application import ApplicationOptions, configure, configure_input
from PyConf.control_flow import CompositeNode

# Use testfiledb for proper conditions/detector setup
options = ApplicationOptions(_enabled=False)
options.set_input_and_conds_from_testfiledb("MiniBrunel_2018_MinBias_FTv4_DIGI")
options.evt_max = 1
options.ntuple_file = "benchmark_nn_field_map.root"
config = configure_input(options)

# Configure detector - only need magnet for extrapolation
if UseDD4Hep:
    dd4hepSvc = config["LHCb::Det::LbDD4hep::DD4hepSvc/LHCb::Det::LbDD4hep::DD4hepSvc"]
    dd4hepSvc.DetectorList = ["/world", "Magnet"]

# ─── Configure RK extrapolator variants ──────────────────────────────────────
# All use CashKarp (default), all have sub-timers enabled.

# 1. Trilinear interpolation (baseline)
trilinear = TrackRungeKuttaExtrapolator(name="Trilinear",
    UseNNFieldMap="none",
    EnableSubTimers=True,
)

# 2. NN scalar SiLU
nn_silu = TrackRungeKuttaExtrapolator(name="NN_SiLU",
    UseNNFieldMap="scalar_silu",
    EnableSubTimers=True,
)

# 3. NN scalar ReLU
nn_relu = TrackRungeKuttaExtrapolator(name="NN_ReLU",
    UseNNFieldMap="scalar_relu",
    EnableSubTimers=True,
)

# 4. NN AVX2 ReLU
nn_avx2 = TrackRungeKuttaExtrapolator(name="NN_AVX2_ReLU",
    UseNNFieldMap="avx2_relu",
    EnableSubTimers=True,
)

# ─── Configure benchmark algorithm ──────────────────────────────────────────
benchmark = TrackExtrapolatorTesterSOA(
    name="BenchmarkNNFieldMap",
    ReferenceExtrapolator=trilinear,
    Extrapolators=[trilinear, nn_silu, nn_relu, nn_avx2],
    InitialZ=3000.0,   # mm - typical VELO exit
    FinalZ=12000.0,     # mm - through full magnet region
    NBins=11,           # 11^3 = 1331 test states
)

# ─── Application setup using configure() ────────────────────────────────────
config.update(configure(options, CompositeNode("TopSeq", [benchmark])))

print("=" * 80)
print("NN Field Map Benchmark")
print("=" * 80)
print("Variants: Trilinear (baseline), NN_SiLU, NN_ReLU, NN_AVX2_ReLU")
print("Sub-timers: ENABLED (RDTSC cycle counts per sub-operation)")
print(f"Output file: benchmark_nn_field_map.root")
print("=" * 80)
