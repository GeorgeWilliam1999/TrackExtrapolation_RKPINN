###############################################################################
# QMTest benchmark configuration for V1 MLP architecture sweep
#
# Benchmarks all exported V1 MLP models against reference extrapolators.
# Models must first be exported to .bin via export_to_c_binary.py.
#
# Usage:
#   gaudirun.py benchmark_v1_models.py
###############################################################################
from Configurables import (
    TrackHerabExtrapolator,
    TrackKiselExtrapolator,
    TrackLinearExtrapolator,
    TrackParabolicExtrapolator,
    TrackRungeKuttaExtrapolator,
)
from TrackExtrapolators.TrackExtrapolatorsConf import TrackMLPExtrapolator
from DDDB.CheckDD4Hep import UseDD4Hep
from PyConf.Algorithms import ExtrapolatorTester
from PyConf.application import ApplicationOptions, configure, configure_input
from PyConf.control_flow import CompositeNode
import os

options = ApplicationOptions(_enabled=False)
options.set_input_and_conds_from_testfiledb("MiniBrunel_2018_MinBias_FTv4_DIGI")
options.evt_max = 1
config = configure_input(options)

if UseDD4Hep:
    dd4hepSvc = config["LHCb::Det::LbDD4hep::DD4hepSvc/LHCb::Det::LbDD4hep::DD4hepSvc"]
    dd4hepSvc.DetectorList = ["/world", "Magnet"]

# Base directory for exported V1 model binaries
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "analysis", "exported_bins"
)

extrapolators = []
ex = ExtrapolatorTester(name="ExtrapolatorTester", Extrapolators=extrapolators)

# Reference extrapolators
extrapolators += [
    TrackRungeKuttaExtrapolator("RK4_Reference"),
    TrackRungeKuttaExtrapolator("BogackiShampine3", RKScheme="BogackiShampine3"),
    TrackRungeKuttaExtrapolator("Verner9", RKScheme="Verner9"),
    TrackKiselExtrapolator("Kisel"),
    TrackHerabExtrapolator("Herab"),
    TrackLinearExtrapolator("Linear"),
    TrackParabolicExtrapolator("Parabolic"),
]

# V1 MLP models (add those that have been exported)
v1_models = [
    "mlp_2x64", "mlp_2x128", "mlp_2x256", "mlp_2x512", "mlp_2x1024",
    "mlp_3x64", "mlp_3x128", "mlp_3x256", "mlp_3x512",
    "mlp_128_64", "mlp_256_128", "mlp_256_256_128",
    "mlp_512_256_128", "mlp_512_512_256",
    "mlp_4x128", "mlp_4x256", "mlp_1024_512_256",
]

for name in v1_models:
    bin_path = os.path.join(MODEL_DIR, f"{name}.bin")
    if os.path.exists(bin_path):
        extrapolators.append(
            TrackMLPExtrapolator(
                f"V1_{name}",
                ModelPath=bin_path,
                Activation="silu",
                NumericalJacobian=True,
            )
        )

config.update(configure(options, CompositeNode("TopSeq", [ex])))
