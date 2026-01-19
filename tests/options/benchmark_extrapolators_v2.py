"""
Minimal benchmark configuration for extrapolators.
Uses simpler setup without full conditions database.
"""

from Configurables import ApplicationMgr, TrackExtrapolatorTesterSOA, NTupleSvc
from Gaudi.Configuration import MessageSvc, INFO

# Configure message service
MessageSvc(OutputLevel=INFO)

# Configure benchmark algorithm
benchmark = TrackExtrapolatorTesterSOA(
    "BenchmarkExtrapolators",
    ReferenceExtrapolator="TrackSTEPExtrapolator",
    Extrapolators=[
        "TrackRungeKuttaExtrapolator",
        "TrackKiselExtrapolator",
        "TrackLinearExtrapolator",
        "TrackParabolicExtrapolator",
        "TrackHerabExtrapolator",
    ],
    InitialZ=4000.0,
    FinalZ=12000.0,
)

# Configure application
ApplicationMgr(
    TopAlg=[benchmark],
    EvtMax=1000,
    HistogramPersistency="ROOT",
    EvtSel="NONE",
)

# Enable ntuple output
NTupleSvc(Output=["FILE1 DATAFILE='benchmark_results.root' OPT='NEW'"])
