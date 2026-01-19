###############################################################################
# (c) Copyright 2000-2022 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
#[=======================================================================[.rst:
Tr/TrackExtrapolators
---------------------
#]=======================================================================]

gaudi_add_module(TrackExtrapolators
    SOURCES
        src/DetailedMaterialLocator.cpp
        src/ExtrapolatorTester.cpp
        src/TrackExtrapolatorTesterSOA.cpp
        src/MaterialLocatorBase.cpp
        src/SimplifiedMaterialLocator.cpp
        src/TrackDistanceExtraSelector.cpp
        src/TrackExtrapolator.cpp
        src/TrackFieldExtrapolatorBase.cpp
        src/TrackHerabExtrapolator.cpp
        src/TrackKiselExtrapolator.cpp
        src/TrackLinearExtrapolator.cpp
        src/TrackMasterExtrapolator.cpp
        src/TrackParabolicExtrapolator.cpp
        src/TrackParametrizedExtrapolator.cpp
        ml_models/src/TrackMLPExtrapolator.cpp
        src/TrackRungeKuttaExtrapolator.cpp
        src/TrackSTEPExtrapolator.cpp
        src/TrackSimpleExtraSelector.cpp
        src/TrackStateProvider.cpp
    LINK
        Boost::headers
        Eigen3::Eigen
        Gaudi::GaudiAlgLib
        Gaudi::GaudiKernel
        GSL::gsl
        LHCb::DetDescLib
        LHCb::LHCbAlgsLib
        LHCb::LHCbKernel
        LHCb::LHCbMathLib
        LHCb::MagnetLib
        LHCb::PartPropLib
        LHCb::TrackEvent
        Rec::TrackFitEvent
        Rec::TrackInterfacesLib
        Rec::TrackKernel
)

gaudi_add_tests(QMTest)
