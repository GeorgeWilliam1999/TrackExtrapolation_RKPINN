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
#ifndef TRACKEXTRAPOLATORS_TRACKEXTRAPOLATOR_H
#define TRACKEXTRAPOLATORS_TRACKEXTRAPOLATOR_H 1

// Include files
// -------------
// from Gaudi
#include "GaudiAlg/GaudiTool.h"

// from TrackInterfaces
#include "TrackInterfaces/ITrackExtrapolator.h"

// from TrackEvent
#include "Event/State.h"
#include "Event/StateVector.h"
#include "Event/States.h"
#include "Event/Track.h"

/** @class TrackExtrapolator TrackExtrapolator.h
 *
 *  A TrackExtrapolator is a base class implementing methods
 *  from the ITrackExtrapolator interface.
 *
 *  @author Edwin Bos (added extrapolation methods)
 *  @date   05/07/2005
 *
 *  @author Eduardo Rodrigues
 *  @date   2004-12-17
 */
class TrackExtrapolator : public extends<GaudiTool, ITrackExtrapolator> {
public:
  /// Standard constructor
  using extends::extends;

  using ITrackExtrapolator::propagate;

  /// Propagate a track to a given z-position
  StatusCode propagate( const LHCb::Track& track, double z, LHCb::State& state, IGeometryInfo const& geometry,
                        const LHCb::Tr::PID                    pid  = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  /// Propagate a track to a given z-position
  StatusCode propagate( const LHCb::Track&, double z, LHCb::StateVector&, IGeometryInfo const& geometry,
                        const LHCb::Tr::PID                    pid  = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  /// Propagate a state to a given z-position
  StatusCode propagate( LHCb::State& state, double z, IGeometryInfo const& geometry,
                        const LHCb::Tr::PID                    pid  = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  /// Propagate a state to a given z-position
  /// Transport matrix is calulated when transMat pointer is not NULL
  StatusCode propagate( LHCb::State& state, double z, Gaudi::TrackMatrix* transMat, IGeometryInfo const& geometry,
                        const LHCb::Tr::PID                    pid  = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  /// Propagate a track to the closest point to the specified point
  StatusCode propagate( const LHCb::Track& track, const Gaudi::XYZPoint& point, LHCb::State& state,
                        IGeometryInfo const& geometry, const LHCb::Tr::PID pid = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  /// Propagate a state to the closest point to the specified point
  StatusCode propagate( LHCb::State& state, const Gaudi::XYZPoint& point, IGeometryInfo const& geometry,
                        const LHCb::Tr::PID                    pid  = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  /// Propagate a track to within tolerance of a plane (default = 10 microns)
  StatusCode propagate( const LHCb::Track& track, const Gaudi::Plane3D& plane, LHCb::State& state,
                        IGeometryInfo const& geometry, double tolerance = 0.01,
                        const LHCb::Tr::PID                    pid  = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  /// Propagate a state to within tolerance of a plane (default = 10 microns)
  StatusCode propagate( LHCb::State& state, const Gaudi::Plane3D& plane, IGeometryInfo const& geometry,
                        double tolerance = 0.01, const LHCb::Tr::PID pid = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  LHCb::Event::v3::States propagate( const LHCb::Event::v3::States& states, double zNew, IGeometryInfo const& geometry,
                                     const LHCb::Tr::PID                    pid  = LHCb::Tr::PID::Pion(),
                                     const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

private:
  /// Maximum number of steps in propagation to a plane
  Gaudi::Property<int> m_maxIter{ this, "Iterations", 5 };

  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_impossible{
      this, "Cannot propagate state to Z at given point. See debug for details", 0 };
  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_parallel{ this, "State parallel to plane!" };
  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_propfail{
      this, "Failed to propagate to given z. See debug for details", 0 };
  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_tolerance{ this,
                                                                     "Failed to propagate to plane within tolerance." };
};
#endif // TRACKEXTRAPOLATORS_TRACKEXTRAPOLATOR_H
