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

// from Gaudi
#include "GaudiAlg/GaudiTool.h"
#include "GaudiKernel/ToolHandle.h"

// from TrackInterfaces
#include "TrackInterfaces/ITrackExtrapolator.h"
#include "TrackInterfaces/ITrackInterpolator.h"
#include "TrackInterfaces/ITrackStateProvider.h"

// from TrackEvent
#include "Event/State.h"
#include "Event/StateParameters.h"
#include "Event/StateVector.h"
#include "Event/Track.h"
#include "Event/TrackFitResult.h"

// from TrackKernel
#include "LHCbMath/Similarity.h"
#include "TrackKernel/TrackFunctors.h"
#include "TrackKernel/TrackTraj.h"

// STL
#include <cstdint>
#include <deque>
#include <memory>
#include <numeric>
#include <unordered_map>

// boost
#include "boost/functional/hash.hpp"

/** @class TrackStateProvider TrackStateProvider.h
 *
 *  A TrackStateProvider is a base class implementing methods
 *  from the ITrackStateProvider interface.
 *
 *  @author Wouter Hulsbergen
 *  @date   14/08/2010
 **/

namespace {

  /// Type for cache key
  using TkCacheKey = std::uintptr_t;

  /// compare states by Z position
  inline constexpr auto compareStateZ = []( const LHCb::State* lhs, const LHCb::State* rhs ) {
    return lhs->z() < rhs->z();
  };

  /// Create the key for a given track
  inline TkCacheKey trackID( const LHCb::Track& track ) {
    TkCacheKey tkid = TkCacheKey( &track );
    if ( track.fitResult() ) { boost::hash_combine( tkid, track.fitResult() ); }
    if ( track.parent() ) { boost::hash_combine( tkid, track.parent() ); }
    // How many IDs are needed to 'guarantee' uniqueness in hash ??
    for ( const auto id : track.lhcbIDs() ) { boost::hash_combine( tkid, id.lhcbID() ); }
    // if ( !track.lhcbIDs().empty() ) {
    //  boost::hash_combine( tkid, track.lhcbIDs().front().lhcbID() );
    //  boost::hash_combine( tkid, track.lhcbIDs().back().lhcbID() );
    //}
    return tkid;
  }

  // The TrackCache is basically just an 'extendable' TrackTraj
  class TrackCache final : public LHCb::TrackTraj {

  private:
    const LHCb::Track*      m_track = nullptr;
    std::deque<LHCb::State> m_ownedstates;
    double                  m_zFirstMeasurement{ -9999 };

  public:
    /// Constructor from a track
    TrackCache( const LHCb::Track& track ) : LHCb::TrackTraj( track ), m_track( &track ) {
      const auto state    = track.stateAt( LHCb::State::Location::FirstMeasurement );
      m_zFirstMeasurement = ( state ? state->z() : -9999 );
    }

    /// get the track
    const LHCb::Track& track() const { return *m_track; }

    /// return z position of state at first measurement
    double zFirstMeasurement() const noexcept { return m_zFirstMeasurement; }

    /// insert a new state in the container with states
    const LHCb::State* insertState( std::ptrdiff_t pos, LHCb::State&& state );

    /// return the states (work around keyword protected)
    auto& states() noexcept { return refStates(); }

    /// return the states (work around keyword protected)
    const auto& states() const noexcept { return refStates(); }

    /// return number of owned states
    std::size_t numOwnedStates() const noexcept { return m_ownedstates.size(); }
  };

  const LHCb::State* TrackCache::insertState( std::ptrdiff_t pos, LHCb::State&& o_state ) {

    // take ownership of state
    auto& state = m_ownedstates.emplace_back( std::move( o_state ) );

    // get the vector with states
    auto& refstates = refStates();

    // insert state
    refstates.insert( std::next( refstates.begin(), pos ), &state );
    assert( std::is_sorted( refstates.begin(), refstates.end(), compareStateZ ) );

    // update the range of the trajectory
    Trajectory::setRange( refstates.front()->z(), refstates.back()->z() );

    // invalidate the cache
    TrackTraj::invalidateCache();

    return &state;
  }

} // namespace

class TrackStateProvider : public extends<GaudiTool, ITrackStateProvider> {

public:
  /// Standard constructor
  TrackStateProvider( const std::string& type, const std::string& name, const IInterface* parent );

  /// Compute the state of the track at position z.  The third
  /// argument is the tolerance: if an existing state is found within
  /// a z-distance 'tolerance', that state is returned.
  /// If there are 'fitnodes' on the track (e.g. in Brunel), this
  /// method will use interpolation. If there are no fitnodes (e.g. in
  /// DaVinci) the method will use extrapolation. In that case the
  /// answer is only correct outside the measurement range.
  StatusCode state( LHCb::State& astate, const LHCb::Track& track, double z, IGeometryInfo const& geometry,
                    double ztolerance ) const override;

  /// Compute state using cached trajectory
  StatusCode stateFromTrajectory( LHCb::State& state, const LHCb::Track& track, double z,
                                  IGeometryInfo const& geometry ) const override {
    const auto traj = trajectory( track, geometry );
    if ( traj ) { state = traj->state( z ); }
    return ( traj ? StatusCode::SUCCESS : StatusCode::FAILURE );
  }

  /// Retrieve the cached trajectory
  const LHCb::TrackTraj* trajectory( const LHCb::Track& track, IGeometryInfo const& geometry ) const override {
    return &cache( track, geometry );
  }

  /// Clear the cache
  void clearCache() const override;

  /// Clear the cache for a particular track
  void clearCache( const LHCb::Track& track ) const override;

private:
  /// Type for cache
  using TrackCaches = std::unordered_map<TkCacheKey, TrackCache>;

private:
  StatusCode computeState( const TrackCache& tc,    //
                           const double      z,     //
                           LHCb::State&      state, //
                           std::ptrdiff_t position, IGeometryInfo const& geometry ) const;

  const LHCb::State* addState( TrackCache& tc, double z, IGeometryInfo const& geometry,
                               LHCb::State::Location loc      = LHCb::State::Location::LocationUnknown,
                               std::ptrdiff_t        position = -1 ) const;

  /// Get the track cache from the event store
  TrackCaches& trackcache() const {
    // auto* obj = m_caches.getIfExists();
    // return const_cast<TrackCaches&>( obj ? *obj : *( m_caches.put( TrackCaches{} ) ) );
    static constexpr auto loc = "TrackStateProviderCache";
    using CacheTES            = AnyDataWrapper<TrackCaches>;
    auto d                    = getIfExists<CacheTES>( loc );
    if ( !d ) {
      d = new CacheTES( TrackCaches{} );
      put( d, loc );
    }
    return d->getData();
  }

  /// Create a cache entry
  TrackCache createCacheEntry( const TkCacheKey key, const LHCb::Track& track, IGeometryInfo const& geometry ) const;

  /// Retrieve a cache entry
  TrackCache& cache( const LHCb::Track& track, IGeometryInfo const& geometry ) const;

private:
  // mutable DataObjectHandle<AnyDataWrapper<TrackCaches>> m_caches{this, Gaudi::DataHandle::Writer, "CacheLocation",
  //                                                               "TrackStateProviderCache"};

  ToolHandle<ITrackExtrapolator> m_extrapolator{ "TrackMasterExtrapolator", this };
  ToolHandle<ITrackInterpolator> m_interpolator{ "TrackInterpolator", this };

  Gaudi::Property<bool>   m_applyMaterialCorrections{ this, "ApplyMaterialCorrections", true };
  Gaudi::Property<double> m_linearPropagationTolerance{ this, "LinearPropagationTolerance", 1.0 * Gaudi::Units::mm };
  Gaudi::Property<bool>   m_cacheStatesOnDemand{ this, "CacheStatesOnDemand", false };

  mutable Gaudi::Accumulators::Counter<> m_addedNOnDemandStates{ this, "AddedNumberOnDemandStates" };
};

/**********************************************************************************************/

DECLARE_COMPONENT( TrackStateProvider )

//=============================================================================
// TrackStateProvider constructor.
//=============================================================================
TrackStateProvider::TrackStateProvider( const std::string& type, //
                                        const std::string& name, //
                                        const IInterface*  parent )
    : base_class( type, name, parent ) {
  declareProperty( "Extrapolator", m_extrapolator );
  declareProperty( "Interpolator", m_interpolator );
}

//=============================================================================
// Clear cache
//=============================================================================
void TrackStateProvider::clearCache() const {
  auto& tc = trackcache();
  if ( msgLevel( MSG::DEBUG ) ) { debug() << "Clearing cache. Size is " << tc.size() << "." << endmsg; }
  tc.clear();
}

//=============================================================================
// Clear cache for a given track
//=============================================================================
void TrackStateProvider::clearCache( const LHCb::Track& track ) const {
  if ( msgLevel( MSG::DEBUG ) ) { debug() << "Clearing cache for track: key=" << track.key() << endmsg; }
  auto& tc = trackcache();
  auto  it = tc.find( trackID( track ) );
  if ( it != tc.end() ) { tc.erase( it ); }
}

//=============================================================================
// get a state at a particular z position, within given tolerance
//=============================================================================

StatusCode TrackStateProvider::state( LHCb::State& state, const LHCb::Track& track, double z,
                                      IGeometryInfo const& geometry, double ztolerance ) const {

  auto& tc = cache( track, geometry );

  // locate the closest state with lower_bound, but cache the
  // insertion point, because we need that multiple times
  state.setZ( z );
  auto               position = std::lower_bound( tc.states().cbegin(), tc.states().cend(), &state, compareStateZ );
  const LHCb::State* closeststate( nullptr );
  if ( position == tc.states().cend() ) {
    closeststate = tc.states().back();
  } else if ( position == tc.states().cbegin() ) {
    closeststate = tc.states().front();
  } else {
    auto prev = std::prev( position );
    // assert( z - (*prev)->z()>=0) ;
    // assert( (*position)->z() - z>=0) ;
    closeststate = ( z - ( *prev )->z() < ( *position )->z() - z ? *prev : *position );
  }

  const auto absdz = std::abs( z - closeststate->z() );
  if ( absdz > std::max( ztolerance, TrackParameters::propagationTolerance ) ) {
    if ( absdz > m_linearPropagationTolerance ) {
      if ( !m_cacheStatesOnDemand ) {
        return computeState( tc, z, state, std::distance( tc.states().cbegin(), position ), geometry );
      }
      const auto newstate = addState( tc, z, geometry, LHCb::State::Location::LocationUnknown,
                                      std::distance( tc.states().cbegin(), position ) );
      ++m_addedNOnDemandStates;
      if ( !newstate ) return StatusCode::FAILURE;
      state = *newstate;
    } else {
      // if we are really close, we'll just use linear extrapolation and do not cache the state.
      state = *closeststate;
      state.linearTransportTo( z );
    }
  } else {
    state = *closeststate;
  }
  return StatusCode::SUCCESS;
}

//=============================================================================
// add a state to the cache of a given track
//=============================================================================

const LHCb::State* TrackStateProvider::addState( TrackCache& tc, double z, IGeometryInfo const& geometry,
                                                 LHCb::State::Location loc, std::ptrdiff_t position ) const {
  auto state = LHCb::State{ loc };
  state.setZ( z );
  if ( position < 0 ) {
    position = std::distance( tc.states().cbegin(),
                              std::lower_bound( tc.states().cbegin(), tc.states().cend(), &state, compareStateZ ) );
  }
  const auto sc = computeState( tc, z, state, position, geometry );
  if ( !sc ) return nullptr;
  state.setLocation( loc );
  if ( msgLevel( MSG::DEBUG ) ) {
    debug() << "Adding state to track cache: key=" << tc.track().key() << " " << z << " " << loc << endmsg;
  }
  return tc.insertState( position, std::move( state ) );
}

StatusCode TrackStateProvider::computeState( const TrackCache& tc, const double z, LHCb::State& state,
                                             std::ptrdiff_t position, IGeometryInfo const& geometry ) const {
  // in brunel, we simply use the interpolator. in davinci, we use the
  // extrapolator, and we take some control over material corrections.
  const auto& track = tc.track();
  const auto* fit   = fitResult( track );
  if ( fit && !fit->nodes().empty() && track.fitStatus() == LHCb::Track::FitStatus::Fitted ) {
    return m_interpolator->interpolate( track, z, state, geometry );
  }

  // locate the states surrounding this z position
  const auto& refstates = tc.states();
  state.setZ( z );
  auto it                       = std::next( refstates.begin(), position );
  bool applyMaterialCorrections = false;
  if ( it == refstates.end() ) {
    state = *refstates.back();
  } else if ( it == refstates.begin() || z < tc.zFirstMeasurement() ) {
    state = **it;
    // if we extrapolate from ClosestToBeam, we don't apply mat corrections.
    applyMaterialCorrections =
        ( state.location() != LHCb::State::Location::ClosestToBeam && m_applyMaterialCorrections );
  } else {
    // take the closest state.
    auto prev = std::prev( it );
    state     = std::abs( ( **it ).z() - z ) < std::abs( ( **prev ).z() - z ) ? **it : **prev;
  }

  if ( msgLevel( MSG::DEBUG ) ) debug() << "Extrapolating to z = " << z << " from state at z= " << state.z() << endmsg;

  if ( applyMaterialCorrections ) return m_extrapolator->propagate( state, z, geometry );

  LHCb::StateVector  statevec( state.stateVector(), state.z() );
  Gaudi::TrackMatrix transmat;

  auto sc = m_extrapolator->propagate( statevec, z, geometry, &transmat );
  if ( sc.isSuccess() ) {
    state.setState( statevec );
    state.setCovariance( LHCb::Math::Similarity( transmat, state.covariance() ) );
  }
  return sc;
}

//=============================================================================
// retrieve the cache for a given track
//=============================================================================

TrackCache TrackStateProvider::createCacheEntry( TkCacheKey key, const LHCb::Track& track,
                                                 IGeometryInfo const& geometry ) const {

  if ( msgLevel( MSG::DEBUG ) ) {
    debug() << "Creating track cache for track: key=" << track.key() << " hashID=" << key << " "
            << track.states().size() << endmsg;
  }

  // create a new entry in the cache
  TrackCache tc{ track };

  // make sure downstream tracks have a few ref states before the first measurement.
  if ( track.type() == LHCb::Track::Types::Downstream ) {
    for ( const auto& loc : { std::pair{ LHCb::State::Location::EndRich1, StateParameters::ZEndRich1 },
                              std::pair{ LHCb::State::Location::BegRich1, StateParameters::ZBegRich1 },
                              std::pair{ LHCb::State::Location::EndVelo, StateParameters::ZEndVelo } } ) {
      if ( !track.stateAt( loc.first ) ) { addState( tc, loc.second, geometry, loc.first ); }
    }
  }

  // make sure all tracks (incl. Downstream) get assigned a state at
  // the beamline. this is useful for the trajectory approximation.
  if ( ( track.hasVelo() || track.hasUT() ) && track.firstState().location() != LHCb::State::Location::ClosestToBeam &&
       !track.stateAt( LHCb::State::Location::ClosestToBeam ) ) {
    // compute poca of first state with z-axis
    const auto& vec = track.firstState().stateVector();
    // check on division by zero (track parallel to beam line!)
    auto t2 = vec[2] * vec[2] + vec[3] * vec[3];
    if ( t2 > 1e-12 ) {
      auto dz = -( vec[0] * vec[2] + vec[1] * vec[3] ) / t2;
      // don't add the state if it is too close
      if ( dz < -10 * Gaudi::Units::cm ) {
        auto z = track.firstState().z() + dz;
        if ( z > -100 * Gaudi::Units::cm ) { // beginning of velo
          addState( tc, z, geometry, LHCb::State::Location::ClosestToBeam );
        }
      }
    }
  }

  // On turbo 2015/2016 there are only two states on the track, namely
  // at the beamline and at rich2. For the trajectory approximation
  // used in DTF for Long-Long Ks, this is not enough. Add a state at
  // the end of the Velo.
  if ( !track.isVeloBackward() && track.hasVelo() && !track.stateAt( LHCb::State::Location::FirstMeasurement ) ) {
    addState( tc, StateParameters::ZEndVelo, geometry, LHCb::State::Location::EndVelo );
  }

  return tc;
}

//=============================================================================
// retrieve TrackCache entry for a given track (from the cache)
//=============================================================================
TrackCache& TrackStateProvider::cache( const LHCb::Track& track, IGeometryInfo const& geometry ) const {
  // get the cache from the stack
  const auto key = trackID( track );
  auto&      tc  = trackcache();
  auto       it  = tc.find( key );
  if ( it == tc.end() ) {
    auto ret = tc.emplace( key, createCacheEntry( key, track, geometry ) );
    it       = ret.first;
  }
  return it->second;
}
