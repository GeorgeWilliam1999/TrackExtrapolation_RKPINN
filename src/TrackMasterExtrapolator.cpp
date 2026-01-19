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
#include "DetDesc/ILVolume.h"
#include "DetDesc/Material.h"
#include "Event/TrackParameters.h"
#include "GaudiKernel/ToolHandle.h"
#include "TrackExtrapolator.h"
#include "TrackInterfaces/IMaterialLocator.h"
#include "TrackInterfaces/ITrackExtraSelector.h"
#include "TrackKernel/CubicStateInterpolationTraj.h"
#include <any>
#include <cmath>
#include <optional>

using namespace Gaudi;
using namespace Gaudi::Units;
using namespace LHCb;

/** @class TrackMasterExtrapolator
 *
 *  A TrackMasterExtrapolator is a ITrackExtrapolator
 *  which calls the other extrapolators to do the extrapolating.
 *  It takes into account:
 *  @li Detector Material (multiple scattering , energy loss)
 *  @li Deals with electrons
 *  @li Checks the input state vector
 *  @li The actual extrapolation is chosen by the extrapolator selector \
 *       m_extraSelector
 *
 *  @author Edwin Bos (added extrapolation methods)
 *  @date   05/07/2005
 *  @author Jose A. Hernando
 *  @date   15-03-05
 *  @author Matt Needham
 *  @date   22-04-2000
 */

class TrackMasterExtrapolator : public TrackExtrapolator {

public:
  using TrackExtrapolator::TrackExtrapolator;

  StatusCode initialize() override {
    return TrackExtrapolator::initialize().andThen( [&] {
      m_useMaterial = ( m_applyMultScattCorr || m_applyEnergyLossCorr || m_applyElectronEnergyLossCorr );
#ifdef USE_DD4HEP
      if ( m_useMaterial ) {
        warning() << "TransportSvc is currently incompatible with DD4HEP. "
                  << "Disabling its use and thus any material corrections." << endmsg;
        warning() << "See https://gitlab.cern.ch/lhcb/Rec/-/issues/326 for more details" << endmsg;
        m_useMaterial = false;
      }
#endif
      if ( !m_useMaterial ) { m_materialLocator.disable(); }
    } );
  }

public:
  using TrackExtrapolator::propagate;

  /// Propagate a state vector from zOld to zNew
  /// Transport matrix is calulated when transMat pointer is not NULL
  StatusCode propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew, Gaudi::TrackMatrix* transMat,
                        IGeometryInfo const& geometry, LHCb::Tr::PID pid = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  /// Propagate a state to a given z-position
  /// Transport matrix is calulated when transMat pointer is not NULL
  StatusCode propagate( LHCb::State& state, double z, Gaudi::TrackMatrix* transMat, IGeometryInfo const& geometry,
                        const LHCb::Tr::PID                    pid  = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

private:
  /// extra selector
  ToolHandle<ITrackExtraSelector> m_extraSelector{ this, "ExtraSelector", "TrackDistanceExtraSelector" };

  /// transport service
  ToolHandle<IMaterialLocator> m_materialLocator{ this, "MaterialLocator", "DetailedMaterialLocator" };

  Gaudi::Property<bool>   m_applyMultScattCorr{ this, "ApplyMultScattCorr",
                                              true }; ///< turn on/off multiple scattering correction
  Gaudi::Property<bool>   m_applyEnergyLossCorr{ this, "ApplyEnergyLossCorr", true }; ///< turn on/off dE/dx correction
  Gaudi::Property<double> m_maxStepSize{ this, "MaxStepSize", 1000. * Gaudi::Units::mm }; ///< maximum length of a step
  Gaudi::Property<double> m_maxSlope{ this, "MaxSlope", 5. }; ///< maximum slope of state vector
  Gaudi::Property<double> m_maxTransverse{ this, "MaxTransverse",
                                           10. * Gaudi::Units::m }; ///< maximum x,y position of state vector
  /// turn on/off electron energy loss corrections
  Gaudi::Property<bool> m_applyElectronEnergyLossCorr{ this, "ApplyElectronEnergyLossCorr", true };
  // Gaudi::Property<double> m_startElectronCorr{ this, "StartElectronCorr", 2500.*mm };  ///< z start for electron
  // energy loss Gaudi::Property<double> m_stopElectronCorr { this, "StopElectronCorr",  9000.*mm };  ///< z start for
  // electron energy loss

  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_absurd{
      this, "Protect against absurd tracks. See debug for details", 1 };
  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_loop{
      this, "Protect against looping tracks. See debug for details", 1 };
  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_extrapFail{ this, "Transport to wall FAILED", 1 };

  /// cache flag indicating if material corrections are enabled
  bool m_useMaterial{ true };
};

DECLARE_COMPONENT( TrackMasterExtrapolator )

//=========================================================================
// Propagate a state vector from zOld to zNew
// Transport matrix is calulated when transMat pointer is not NULL
// Note: energy loss correction is _NOT_ applied.
//=========================================================================
StatusCode TrackMasterExtrapolator::propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew,
                                               Gaudi::TrackMatrix* transMat, IGeometryInfo const& geometry,
                                               const LHCb::Tr::PID                    pid,
                                               const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  // Gaudi::Property<double> m_shortDist { this,  "shortDist", 100.0*Gaudi::Units::mm };
  // return std::abs(zEnd-zStart) < m_shortDist ?
  //        m_shortDistanceExtrapolator : m_longDistanceExtrapolator;
  const auto thisExtrapolator = m_extraSelector->select( zOld, zNew );
  return thisExtrapolator->propagate( stateVec, zOld, zNew, transMat, geometry, pid, grid );
}

//=========================================================================
//  Main method: Extrapolate a State
//=========================================================================
StatusCode TrackMasterExtrapolator::propagate( LHCb::State& state, double zNew, Gaudi::TrackMatrix* transMat,
                                               IGeometryInfo const& geometry, LHCb::Tr::PID pid,
                                               const LHCb::Magnet::MagneticFieldGrid* grid ) const {

  // Create transport update matrix. The reason to make a pointer to a
  // local object (rather than just create it with new) is all the
  // intermediate returns.
  TrackMatrix  updateMatrix = ROOT::Math::SMatrixIdentity();
  TrackMatrix* upMat        = nullptr;
  if ( transMat ) {
    *transMat = ROOT::Math::SMatrixIdentity();
    upMat     = &updateMatrix;
  }

  StatusCode sc{ StatusCode::SUCCESS };
  // check if not already at required z position
  const auto zStart = state.z();
  if ( std::abs( zNew - zStart ) < TrackParameters::propagationTolerance ) {
    if ( msgLevel( MSG::DEBUG ) ) { debug() << "already at required z position" << endmsg; }
    return sc;
  }

  const auto nbStep    = (int)( std::abs( zNew - zStart ) / m_maxStepSize ) + 1;
  const auto zStep     = ( zNew - zStart ) / nbStep;
  size_t     nWallsTot = 0;

  auto materialCache = ( m_useMaterial ? m_materialLocator->createCache() : std::any{} );

  std::optional<LHCb::State> stateAtOrigin;

  if ( msgLevel( MSG::VERBOSE ) ) {
    verbose() << "state_in = " << state << "\nz_out = " << zNew << "num steps = " << nbStep << endmsg;
  }

  for ( int step = 0; nbStep > step; ++step ) {
    const auto&     tX = state.stateVector();
    const XYZPoint  start( tX[0], tX[1], state.z() ); // Initial position
    const XYZVector vect( tX[2] * zStep, tX[3] * zStep, zStep );

    // protect against vertical or looping tracks
    if ( std::abs( start.x() ) > m_maxTransverse ) {
      if ( msgLevel( MSG::DEBUG ) ) {
        debug() << "Protect against absurd tracks: x=" << start.x() << " (max " << m_maxTransverse << " allowed)."
                << endmsg;
      }
      ++m_absurd;
      return StatusCode::FAILURE;
    }
    if ( std::abs( start.y() ) > m_maxTransverse ) {
      if ( msgLevel( MSG::DEBUG ) ) {
        debug() << "Protect against absurd tracks: y=" << start.y() << " (max " << m_maxTransverse << " allowed)."
                << endmsg;
      }
      ++m_absurd;
      return StatusCode::FAILURE;
    }
    if ( std::abs( state.tx() ) > m_maxSlope ) {
      if ( msgLevel( MSG::DEBUG ) ) {
        debug() << "Protect against looping tracks: tx=" << state.tx() << " (max " << m_maxSlope << " allowed)."
                << endmsg;
      }
      ++m_loop;
      return StatusCode::FAILURE;
    }
    if ( std::abs( state.ty() ) > m_maxSlope ) {
      if ( msgLevel( MSG::DEBUG ) ) {
        debug() << "Protect against looping tracks: ty=" << state.ty() << " (max " << m_maxSlope << " allowed). "
                << endmsg;
      }
      ++m_loop;
      return StatusCode::FAILURE;
    }

    // propagate the state, without any material corrections:
    const auto zorigin = state.z();
    const auto ztarget = zorigin + zStep;

    if ( m_useMaterial ) { stateAtOrigin = state; }

    const auto thisExtrapolator = m_extraSelector->select( zorigin, ztarget );
    sc                          = thisExtrapolator->propagate( state, ztarget, upMat, geometry, pid, grid );

    // check for success
    if ( sc.isFailure() ) {
      if ( msgLevel( MSG::DEBUG ) ) {
        debug() << "Transport to " << ztarget << "using " + thisExtrapolator->name() << " FAILED" << endmsg;
      }
      return sc;
    }

    // update f
    if ( transMat ) {
      TrackMatrix tempMatrix = *transMat;
      *transMat              = updateMatrix * tempMatrix;
    }

    // now apply material corrections
    if ( m_useMaterial ) {
      assert( stateAtOrigin.has_value() );
      LHCb::CubicStateInterpolationTraj traj( stateAtOrigin.value(), state );
      IMaterialLocator::Intersections   intersections = m_materialLocator->intersect( traj, materialCache, geometry );
      if ( intersections.size() > 0 ) {
        nWallsTot += intersections.size();
        m_materialLocator->applyMaterialCorrections( state, intersections, zorigin, pid, m_applyMultScattCorr,
                                                     m_applyEnergyLossCorr || m_applyElectronEnergyLossCorr );
      }
    }
  } // loop over steps

  if ( msgLevel( MSG::VERBOSE ) ) {
    verbose() << "state_out = " << state << std::endl << "number of walls = " << nWallsTot << endmsg;
  }

  return sc;
}
