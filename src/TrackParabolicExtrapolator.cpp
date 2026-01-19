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
#include "Core/FloatComparison.h"
#include "Event/TrackParameters.h"
#include "GaudiKernel/PhysicalConstants.h"
#include "GaudiKernel/Point3DTypes.h"
#include "GaudiKernel/Vector3DTypes.h"
#include "TrackFieldExtrapolatorBase.h"
#include "gsl/gsl_math.h"
#include "gsl/gsl_poly.h"
#include <string>

using namespace Gaudi::Units;
using namespace LHCb;
using namespace Gaudi;

/** @class TrackParabolicExtrapolator TrackParabolicExtrapolator.h \
 *         "TrackParabolicExtrapolator.h"
 *
 *  A TrackParabolicExtrapolator is a ITrackExtrapolator that does a transport
 *  using a parabolic expansion of the trajectory. It doesn't take into
 *  account Multiple Scattering.
 *
 *  @author Edwin Bos (added extrapolation methods)
 *  @date   05/07/2005
 *  @author Jose A. Hernando (13-03-2005)
 *  @author Matt Needham
 *  @date   22-04-2000
 */

class TrackParabolicExtrapolator : public TrackFieldExtrapolatorBase {

public:
  /// Constructor
  using TrackFieldExtrapolatorBase::propagate;
  using TrackFieldExtrapolatorBase::TrackFieldExtrapolatorBase;

  /// Propagate a state vector from zOld to zNew
  /// Transport matrix is calulated when transMat pointer is not NULL
  StatusCode propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew, Gaudi::TrackMatrix* transMat,
                        IGeometryInfo const& geometry, const LHCb::Tr::PID pid = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  /// Propagate a state to the closest position to the specified point
  StatusCode propagate( LHCb::State& state, const Gaudi::XYZPoint& point, IGeometryInfo const& geometry,
                        const LHCb::Tr::PID                    pid  = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

private:
  /// update transport matrix
  void updateTransportMatrix( const double dz, Gaudi::TrackVector& stateVec, Gaudi::TrackMatrix& transMat,
                              const Gaudi::XYZVector& B, double ax, double ay ) const;
};

DECLARE_COMPONENT( TrackParabolicExtrapolator )

//=============================================================================
// Propagate a state vector from zOld to zNew
// Transport matrix is calulated when transMat pointer is not NULL
//=============================================================================
StatusCode TrackParabolicExtrapolator::propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew,
                                                  Gaudi::TrackMatrix* transMat, IGeometryInfo const&,
                                                  const LHCb::Tr::PID /*pid*/,
                                                  const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  // Bail out if already at destination
  const double dz = zNew - zOld;
  if ( std::abs( dz ) < TrackParameters::propagationTolerance ) {
    if ( msgLevel( MSG::DEBUG ) ) debug() << "already at required z position" << endmsg;
    // Reset the transport matrix
    if ( transMat ) *transMat = ROOT::Math::SMatrixIdentity();
    return StatusCode::SUCCESS;
  }

  // get the B field at midpoint
  const double     xMid = stateVec[0] + ( 0.5 * stateVec[2] * dz );
  const double     yMid = stateVec[1] + ( 0.5 * stateVec[3] * dz );
  XYZPoint         P( xMid, yMid, zOld + ( 0.5 * dz ) );
  Gaudi::XYZVector m_B = fieldVector( grid, P );

  // to save some typing...
  const double Tx   = stateVec[2];
  const double Ty   = stateVec[3];
  const double nTx2 = 1.0 + Tx * Tx;
  const double nTy2 = 1.0 + Ty * Ty;
  const double norm = std::sqrt( nTx2 + nTy2 - 1.0 );

  // calculate the A factors
  double m_ax = norm * ( Ty * ( Tx * m_B.x() + m_B.z() ) - ( nTx2 * m_B.y() ) );
  double m_ay = norm * ( -Tx * ( Ty * m_B.y() + m_B.z() ) + ( nTy2 * m_B.x() ) );

  const double fac  = eplus * c_light * dz;
  const double fact = fac * stateVec[4];

  // Update the state parameters (exact extrapolation)
  stateVec[0] += dz * ( Tx + 0.5 * m_ax * fact );
  stateVec[1] += dz * ( Ty + 0.5 * m_ay * fact );
  stateVec[2] += m_ax * fact;
  stateVec[3] += m_ay * fact;

  if ( transMat ) { updateTransportMatrix( dz, stateVec, *transMat, m_B, m_ax, m_ay ); }

  return StatusCode::SUCCESS;
}

//=============================================================================
// Propagate a state to the closest position to the specified point
//=============================================================================
StatusCode TrackParabolicExtrapolator::propagate( State& state, const Gaudi::XYZPoint& point,
                                                  IGeometryInfo const& geometry, const LHCb::Tr::PID pid,
                                                  const LHCb::Magnet::MagneticFieldGrid* grid ) const {

  // Check whether not already at reference point
  XYZPoint  P    = state.position();
  XYZVector diff = point - P;
  if ( diff.R() < TrackParameters::propagationTolerance ) { return StatusCode::SUCCESS; }

  Gaudi::XYZVector m_B = fieldVector( grid, P + 0.5 * diff );

  // The distance between the reference point and a point on the parabola
  // can be minimized by taking the derivative wrt Z and equal that to zero.
  // This implies solving a cubic equation, resulting in 1 or 3 solutions.

  double Tx    = state.tx();
  double Ty    = state.ty();
  double nTx2  = 1.0 + Tx * Tx;
  double nTy2  = 1.0 + Ty * Ty;
  double norm  = std::sqrt( nTx2 + nTy2 - 1.0 );
  double m_ax  = norm * ( Ty * ( Tx * m_B.x() + m_B.z() ) - ( nTx2 * m_B.y() ) );
  double m_ay  = norm * ( -Tx * ( Ty * m_B.y() + m_B.z() ) + ( nTy2 * m_B.x() ) );
  double varA  = 0.5 * m_ax * state.qOverP() * eplus * c_light;
  double varB  = 0.5 * m_ay * state.qOverP() * eplus * c_light;
  double alpha = 2. * ( varA * varA + varB * varB );
  double beta  = 3. * ( Tx * varA + Ty * varB );

  double gamma =
      2. * P.x() * varA + Tx * Tx + 1. - 2. * point.x() * varA + 2. * P.y() * varB + Ty * Ty - 2. * point.y() * varB;

  double delta = P.x() * Tx - point.x() * Tx + P.y() * Ty - point.y() * Ty + P.z() - point.z();

  // Create parameters in which to store the solutions (zNew = zOld+sol)
  double so1 = 999.;
  double so2 = 999.;
  double so3 = 999.;

  gsl_poly_solve_cubic( beta / alpha, gamma / alpha, delta / alpha, &so1, &so2, &so3 );

  // Choose the solution closest to the present position
  bool use1 = ( so1 >= 0. && !LHCb::essentiallyEqual( so1, 999. ) );
  return TrackExtrapolator::propagate( state, P.z() + ( use1 ? so1 : so3 ), geometry, pid, grid );
}

//=============================================================================
// Update the transport matrix
//=============================================================================
void TrackParabolicExtrapolator::updateTransportMatrix( const double dz, Gaudi::TrackVector& stateVec,
                                                        Gaudi::TrackMatrix& transMat, const Gaudi::XYZVector& m_B,
                                                        double m_ax, double m_ay ) const {
  // to save some typing...
  double Tx    = stateVec[2];
  double Ty    = stateVec[3];
  double norm2 = 1. + Tx * Tx + Ty * Ty;
  double norm  = std::sqrt( norm2 );

  // calculate derivatives of Ax, Ay
  double dAx_dTx = ( Tx * m_ax / norm2 ) + norm * ( Ty * m_B.x() - ( 2. * Tx * m_B.y() ) );
  double dAx_dTy = ( Ty * m_ax / norm2 ) + norm * ( Tx * m_B.x() + m_B.z() );
  double dAy_dTx = ( Tx * m_ay / norm2 ) + norm * ( -Ty * m_B.y() - m_B.z() );
  double dAy_dTy = ( Ty * m_ay / norm2 ) + norm * ( -Tx * m_B.y() + ( 2. * Ty * m_B.x() ) );

  // fill transport matrix
  double fac  = eplus * c_light * dz;
  double fact = fac * stateVec[4];

  transMat( 0, 0 ) = 1;
  transMat( 0, 1 ) = 0;
  transMat( 0, 2 ) = dz + 0.5 * dAx_dTx * fact * dz;
  transMat( 0, 3 ) = 0.5 * dAx_dTy * fact * dz;
  transMat( 0, 4 ) = 0.5 * m_ax * fac * dz;

  transMat( 1, 0 ) = 0;
  transMat( 1, 1 ) = 1;
  transMat( 1, 2 ) = 0.5 * dAy_dTx * fact * dz;
  transMat( 1, 3 ) = dz + 0.5 * dAy_dTy * fact * dz;
  transMat( 1, 4 ) = 0.5 * m_ay * fac * dz;

  transMat( 2, 0 ) = 0;
  transMat( 2, 1 ) = 0;
  transMat( 2, 2 ) = 1.0 + dAx_dTx * fact;
  transMat( 2, 3 ) = dAx_dTy * fact;
  transMat( 2, 4 ) = m_ax * fac;

  transMat( 3, 0 ) = 0;
  transMat( 3, 1 ) = 0;
  transMat( 3, 2 ) = dAy_dTx * fact;
  transMat( 3, 3 ) = 1.0 + dAy_dTy * fact;
  transMat( 3, 4 ) = m_ay * fac;

  transMat( 4, 0 ) = 0;
  transMat( 4, 1 ) = 0;
  transMat( 4, 2 ) = 0;
  transMat( 4, 3 ) = 0;
  transMat( 4, 4 ) = 1;
}
