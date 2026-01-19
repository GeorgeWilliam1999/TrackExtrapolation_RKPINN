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
// Include files

// local
#include "TrackExtrapolator.h"
#include "Event/TrackParameters.h"
#include "TrackKernel/TrackFunctors.h"

#include <math.h>

#include "LHCbMath/GeomFun.h"
#include "LHCbMath/Line.h"
#include "LHCbMath/Similarity.h"

using namespace LHCb;
using namespace Gaudi;

//=============================================================================
// Propagate a track to a given z-position
//=============================================================================
StatusCode TrackExtrapolator::propagate( const Track& track, double z, State& state, IGeometryInfo const& geometry,
                                         const LHCb::Tr::PID pid, const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  // get state closest to z
  state = closestState( track, z );

  // propagate the closest state
  return propagate( state, z, geometry, pid, grid );
}

//=============================================================================
// Propagate a track to a given z-position
//=============================================================================
StatusCode TrackExtrapolator::propagate( const Track& track, double z, StateVector& state,
                                         IGeometryInfo const& geometry, const LHCb::Tr::PID pid,
                                         const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  // get state closest to z
  const State& closest = closestState( track, z );
  state                = LHCb::StateVector( closest.stateVector(), closest.z() );

  // propagate the closest state
  return propagate( state, z, geometry, 0, pid, grid );
}

//=============================================================================
// Propagate a state to a given z-position
//=============================================================================
StatusCode TrackExtrapolator::propagate( State& state, double z, IGeometryInfo const& geometry, const LHCb::Tr::PID pid,
                                         const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  Gaudi::TrackMatrix transMat = ROOT::Math::SMatrixIdentity();
  return propagate( state, z, &transMat, geometry, pid, grid );
}

//=============================================================================
// Propagate a state to a given z-position
// Transport matrix is calulated when transMat pointer is not NULL
//=============================================================================
StatusCode TrackExtrapolator::propagate( State& state, double z, Gaudi::TrackMatrix* tm, IGeometryInfo const& geometry,
                                         const LHCb::Tr::PID pid, const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  Gaudi::TrackMatrix transMat = ROOT::Math::SMatrixIdentity();
  return propagate( state.stateVector(), state.z(), z, &transMat, geometry, pid, grid ).andThen( [&] {
    state.setZ( z );
    state.setCovariance( LHCb::Math::Similarity( transMat, state.covariance() ) );
    if ( tm ) *tm = transMat;
  } );
}

//=============================================================================
// Propagate a track to the closest point to the specified point
//=============================================================================
StatusCode TrackExtrapolator::propagate( Track const& track, Gaudi::XYZPoint const& point, LHCb::State& state,
                                         IGeometryInfo const& geometry, LHCb::Tr::PID pid,
                                         const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  // get state closest to z of point
  state = closestState( track, point.z() );

  // propagate the closest state
  return propagate( state, point.z(), geometry, pid, grid );
}

//=============================================================================
// Propagate a state to the closest point to the specified point
//=============================================================================
StatusCode TrackExtrapolator::propagate( State& state, const Gaudi::XYZPoint& point, IGeometryInfo const&,
                                         const LHCb::Tr::PID, const LHCb::Magnet::MagneticFieldGrid* ) const {
  ++m_impossible;

  if ( msgLevel( MSG::DEBUG ) )
    debug() << " can not propagate state at " << state.z() << " to point at z " << point.z() << endmsg;

  return StatusCode::FAILURE;
}

//=============================================================================
// Propagate a track to within tolerance of a plane (default = 10 microns)
//=============================================================================
StatusCode TrackExtrapolator::propagate( const Track& track, const Gaudi::Plane3D& plane, LHCb::State& state,
                                         IGeometryInfo const& geometry, double tolerance, const LHCb::Tr::PID pid,
                                         const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  // get state closest to the plane
  state = closestState( track, plane );

  // propagate the closest state
  return propagate( state, plane, geometry, tolerance, pid, grid );
}

//=============================================================================
// Propagate a state to within tolerance of a plane (default = 10 microns)
//=============================================================================
StatusCode TrackExtrapolator::propagate( State& state, const Gaudi::Plane3D& plane, IGeometryInfo const& geometry,
                                         double tolerance, const LHCb::Tr::PID pid,
                                         const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  StatusCode      sc = StatusCode::FAILURE;
  Gaudi::XYZPoint intersect;
  int             iter;
  double          distance;
  for ( iter = 0; iter < m_maxIter; ++iter ) {
    Gaudi::Math::Line<Gaudi::XYZPoint, Gaudi::XYZVector> line( state.position(), state.slopes() );
    double                                               dz;
    bool success = Gaudi::Math::intersection( line, plane, intersect, dz );
    if ( !success ) {
      ++m_parallel;
      break;
    }
    distance = ( intersect - line.beginPoint() ).R();

    if ( distance < tolerance ) {
      sc = StatusCode::SUCCESS;
      break;
    } else {
      double ztarget = state.z() + dz;
      sc             = propagate( state, ztarget, geometry, pid, grid );
      if ( sc.isFailure() ) {
        ++m_propfail;
        if ( msgLevel( MSG::DEBUG ) ) debug() << "Failed to propagate to z = " << ztarget << endmsg;
        break;
      }
    }
  }

  if ( iter == m_maxIter ) ++m_tolerance;

  return sc;
}

LHCb::Event::v3::States TrackExtrapolator::propagate( const LHCb::Event::v3::States& states, double zNew,
                                                      IGeometryInfo const& geometry, const LHCb::Tr::PID pid,
                                                      const LHCb::Magnet::MagneticFieldGrid* grid ) const {

  LHCb::Event::v3::States out_states;
  out_states.reserve( states.size() );

  for ( auto state : states.scalar() ) {

    auto input_code = state.get<LHCb::Event::v3::StatesTag::Success>().cast();
    auto is_valid   = StatusCode( input_code ).isSuccess();

    auto s   = state.get<LHCb::Event::v3::StatesTag::State>();
    auto zin = s.z().cast();
    auto x   = s.x().cast();
    auto y   = s.y().cast();
    auto tx  = s.tx().cast();
    auto ty  = s.ty().cast();
    auto qop = s.qOverP().cast();

    LHCb::StateVector sv{ { x, y, tx, ty, qop }, zin };
    LHCb::State       st{ sv };

    auto                  cov       = state.get<LHCb::Event::v3::StatesTag::Covariance>();
    Gaudi::TrackSymMatrix gaudi_cov = Gaudi::TrackSymMatrix();
    gaudi_cov( 0, 0 )               = cov.x_x().cast();
    gaudi_cov( 0, 1 )               = cov.x_y().cast();
    gaudi_cov( 0, 2 )               = cov.x_tx().cast();
    gaudi_cov( 0, 3 )               = cov.x_ty().cast();
    gaudi_cov( 0, 4 )               = cov.x_QoverP().cast();
    gaudi_cov( 1, 1 )               = cov.y_y().cast();
    gaudi_cov( 1, 2 )               = cov.y_tx().cast();
    gaudi_cov( 1, 3 )               = cov.y_ty().cast();
    gaudi_cov( 1, 4 )               = cov.y_QoverP().cast();
    gaudi_cov( 2, 2 )               = cov.tx_tx().cast();
    gaudi_cov( 2, 3 )               = cov.tx_ty().cast();
    gaudi_cov( 2, 4 )               = cov.tx_QoverP().cast();
    gaudi_cov( 3, 3 )               = cov.ty_ty().cast();
    gaudi_cov( 3, 4 )               = cov.ty_QoverP().cast();
    gaudi_cov( 4, 4 )               = cov.QoverP_QoverP().cast();
    st.setCovariance( gaudi_cov );

    auto status = propagate( st, zNew, nullptr, geometry, pid, grid );

    auto out_state = out_states.emplace_back<SIMDWrapper::InstructionSet::Scalar>();
    auto success   = is_valid ? status.getCode() : input_code;

    out_state.field<LHCb::Event::v3::StatesTag::Success>().set( success );
    out_state.field<LHCb::Event::v3::StatesTag::State>().setPosition( st.x(), st.y(), st.z() );
    out_state.field<LHCb::Event::v3::StatesTag::State>().setDirection( st.tx(), st.ty() );
    out_state.field<LHCb::Event::v3::StatesTag::State>().setQOverP( st.qOverP() );

    // extremely stupid for now
    auto out_cov       = st.covariance();
    auto x_x           = out_cov( 0, 0 );
    auto x_y           = out_cov( 0, 1 );
    auto x_tx          = out_cov( 0, 2 );
    auto x_ty          = out_cov( 0, 3 );
    auto x_QoverP      = out_cov( 0, 4 );
    auto y_y           = out_cov( 1, 1 );
    auto y_tx          = out_cov( 1, 2 );
    auto y_ty          = out_cov( 1, 3 );
    auto y_QoverP      = out_cov( 1, 4 );
    auto tx_tx         = out_cov( 2, 2 );
    auto tx_ty         = out_cov( 2, 3 );
    auto tx_QoverP     = out_cov( 2, 4 );
    auto ty_ty         = out_cov( 3, 3 );
    auto ty_QoverP     = out_cov( 3, 4 );
    auto QoverP_QoverP = out_cov( 4, 4 );

    out_state.field<LHCb::Event::v3::StatesTag::Covariance>().set( x_x, x_y, x_tx, x_ty, x_QoverP, y_y, y_tx, y_ty,
                                                                   y_QoverP, tx_tx, tx_ty, tx_QoverP, ty_ty, ty_QoverP,
                                                                   QoverP_QoverP );
  }

  return out_states;
}
//=============================================================================
