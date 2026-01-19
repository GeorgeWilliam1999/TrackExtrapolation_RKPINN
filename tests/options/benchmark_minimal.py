#!/usr/bin/env python
"""
Minimal benchmark - simplified approach that works with local setup.
Generates synthetic track states for benchmarking extrapolators.
"""

from Configurables import (
    ApplicationMgr,
    TrackExtrapolatorTesterSOA,
    NTupleSvc,
    MessageSvc,
    LHCbApp,
)

# Use LHCbApp for proper initialization 
LHCbApp(
    DataType="Upgrade",
    Simulation=True,
    DDDBtag="dddb-20200914",
    CondDBtag="sim-20200513-vc-md100",
)

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
    EvtSel="NONE",
)

# Enable output
NTupleSvc(Output=["FILE1 DATAFILE='benchmark_results.root' OPT='NEW'"])
MessageSvc(OutputLevel=3)  # INFO level
