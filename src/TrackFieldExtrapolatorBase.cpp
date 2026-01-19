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
    if ( m_useGridInterpolation ) {
      m_fieldFunction = []( const FGrid* grid, const Gaudi::XYZPoint& p ) {
        return grid->fieldVectorLinearInterpolation( p );
      };
    } else {
      m_fieldFunction = []( const FGrid* grid, const Gaudi::XYZPoint& p ) {
        return grid->fieldVectorClosestPoint( p );
      };
    }
    if ( msgLevel( MSG::DEBUG ) ) { debug() << "UseGridInterpolation: " << m_useGridInterpolation << endmsg; }
  } );
}
