/*****************************************************************************\
* (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "Event/TrackParameters.h"
#include "TrackExtrapolator.h"

/** @class TrackLinearExtrapolator TrackLinearExtrapolator.h TrackExtrapolators/TrackLinearExtrapolator.h
 *
 *  A TrackLinearExtrapolator is a ITrackExtrapolator which does a 'linear'
 *  (i.e. straight line) extrapolation of a State. It doesn't take into
 *  account MS.
 *
 *  @author Edwin Bos (added extrapolation method)
 *  @date   05/07/2005
 *
 *  @author Eduardo Rodrigues (changes and new features for new track event model)
 *  @date   25/11/2004
 *
 *  @author Rutger van der Eijk
 *  @date   07-04-1999
 */

struct TrackLinearExtrapolator : TrackExtrapolator {

  /// constructor
  using TrackExtrapolator::TrackExtrapolator;

  using TrackExtrapolator::propagate;

  /// Propagate a state vector from zOld to zNew
  /// Transport matrix is calulated when transMat pointer is not NULL
  StatusCode propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew, Gaudi::TrackMatrix* transMat,
                        IGeometryInfo const& geometry, const LHCb::Tr::PID pid = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  /// Propagate a state to the closest position to the specified point
  StatusCode propagate( LHCb::State& state, const Gaudi::XYZPoint& point, IGeometryInfo const& geometry,
                        const LHCb::Tr::PID                    pid  = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;
};

DECLARE_COMPONENT( TrackLinearExtrapolator )

//=============================================================================
// Propagate a state vector from zOld to zNew
// Transport matrix is calulated when transMat pointer is not NULL
//=============================================================================
StatusCode TrackLinearExtrapolator::propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew,
                                               Gaudi::TrackMatrix* transMat, IGeometryInfo const&,
                                               const LHCb::Tr::PID /*pid*/,
                                               const LHCb::Magnet::MagneticFieldGrid* ) const {
  // Bail out if already at destination
  const double dz = zNew - zOld;
  if ( fabs( dz ) < TrackParameters::propagationTolerance ) {
    if ( msgLevel( MSG::DEBUG ) ) debug() << "already at required z position" << endmsg;
    if ( transMat ) *transMat = ROOT::Math::SMatrixIdentity();
    return StatusCode::SUCCESS;
  }

  if ( transMat ) {
    ( *transMat )         = ROOT::Math::SMatrixIdentity();
    ( *transMat )( 0, 2 ) = dz;
    ( *transMat )( 1, 3 ) = dz;
  }

  stateVec[0] += stateVec[2] * dz;
  stateVec[1] += stateVec[3] * dz;

  return StatusCode::SUCCESS;
}

//=============================================================================
// Propagate a State to the closest position to the specified point
//=============================================================================
StatusCode TrackLinearExtrapolator::propagate( LHCb::State& state, const Gaudi::XYZPoint& point,
                                               IGeometryInfo const& geometry, const LHCb::Tr::PID pid,
                                               const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  // Distance = sqrt((x'-x0-Tx*dz)^2+(y'-y0-Ty*dz)^2+(z'-z0-dz)^2)
  // Find dz by solving: d(distance)/dz = 0
  Gaudi::XYZVector slo = state.slopes();
  Gaudi::XYZVector dif = state.position() - point;

  // Remember that slo.Z()==1 by definition
  double zNew = -2 * ( ( dif.X() + dif.Y() + dif.Z() ) / ( slo.X() + slo.Y() + 1 ) );

  // Propagate to the point
  return TrackExtrapolator::propagate( state, zNew, geometry, pid, grid );
}
