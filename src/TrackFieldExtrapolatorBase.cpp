/*****************************************************************************\
* (c) Copyright 2000-2022 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
// Include files
// -------------
// from Gaudi

#include "TrackFieldExtrapolatorBase.h"
#include "FieldMapNNWeights.h"
#include "FieldMapNNReLUWeights.h"

//=============================================================================
// Initialization
//=============================================================================
StatusCode TrackFieldExtrapolatorBase::initialize() {
#ifndef USE_DD4HEP
  // make sure the MagneticFieldSvc is available (to be backward compatible with old job configurations)
  service( "MagneticFieldSvc" );
#endif
  return ConditionAccessorHolder::initialize().andThen( [&] {
    using FGrid = LHCb::Magnet::MagneticFieldGrid;
    const auto& nn = m_nnFieldMap.value();
    if ( nn == "scalar_silu" ) {
      m_fieldFunction = []( const FGrid*, const Gaudi::XYZPoint& p ) {
        float bx, by, bz;
        LHCb::FieldNN::evaluate( float( p.x() ), float( p.y() ), float( p.z() ), bx, by, bz );
        return Gaudi::XYZVector{ bx, by, bz };
      };
      info() << "Using NN field map: scalar_silu" << endmsg;
    } else if ( nn == "scalar_relu" ) {
      m_fieldFunction = []( const FGrid*, const Gaudi::XYZPoint& p ) {
        float bx, by, bz;
        LHCb::FieldNNReLU::evaluate_relu( float( p.x() ), float( p.y() ), float( p.z() ), bx, by, bz );
        return Gaudi::XYZVector{ bx, by, bz };
      };
      info() << "Using NN field map: scalar_relu" << endmsg;
    } else if ( nn == "avx2_relu" ) {
      m_fieldFunction = []( const FGrid*, const Gaudi::XYZPoint& p ) {
        float bx, by, bz;
        LHCb::FieldNNReLU::evaluate_relu_avx2( float( p.x() ), float( p.y() ), float( p.z() ), bx, by, bz );
        return Gaudi::XYZVector{ bx, by, bz };
      };
      info() << "Using NN field map: avx2_relu" << endmsg;
    } else if ( m_useGridInterpolation ) {
      m_fieldFunction = []( const FGrid* grid, const Gaudi::XYZPoint& p ) {
        return grid->fieldVectorLinearInterpolation( p );
      };
    } else {
      m_fieldFunction = []( const FGrid* grid, const Gaudi::XYZPoint& p ) {
        return grid->fieldVectorClosestPoint( p );
      };
    }
    if ( msgLevel( MSG::DEBUG ) ) {
      debug() << "UseGridInterpolation: " << m_useGridInterpolation
              << " UseNNFieldMap: " << m_nnFieldMap << endmsg;
    }
  } );
}
