###############################################################################
# In-stack validation of NNVertexFitExtrapolator (the vertex-fit neural
# surrogate) against TrackRungeKuttaExtrapolator, on the two leg shapes the
# vertex fitters actually request:
#   - leg A: UT band -> vertex basin (backward through the fringe field)
#   - leg D: short intra-basin hop (the DecayTreeFitter re-fetch)
# The grid also contains qop = 0 (neutral) rows, which exercise the tool's
# out-of-domain fallback to the Runge-Kutta extrapolator.
#
# Run:  Rec/run gaudirun.py Tr/TrackExtrapolators/tests/options/benchmark_nn_vertexfit.py
###############################################################################
from Configurables import TrackRungeKuttaExtrapolator
from TrackExtrapolators.TrackExtrapolatorsConf import NNVertexFitExtrapolator
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

extrapsA = []
exA = ExtrapolatorTester(
    name="VFLegA", Extrapolators=extrapsA, InitialZ=2500.0, FinalZ=1200.0
)
extrapsA += [
    TrackRungeKuttaExtrapolator("ReferenceA"),
    NNVertexFitExtrapolator("NNVFA"),
]

extrapsD = []
exD = ExtrapolatorTester(
    name="VFLegD", Extrapolators=extrapsD, InitialZ=1800.0, FinalZ=1600.0
)
extrapsD += [
    TrackRungeKuttaExtrapolator("ReferenceD"),
    NNVertexFitExtrapolator("NNVFD"),
]

config.update(configure(options, CompositeNode("TopSeq", [exA, exD])))
