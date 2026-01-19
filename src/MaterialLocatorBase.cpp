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
#include "MaterialLocatorBase.h"
#include "DetDesc/Material.h"
#include "Event/StateParameters.h"
#include "Event/TrackParameters.h"
#include "Kernel/TrackDefaultParticles.h"
#include "TrackKernel/CubicStateVectorInterpolationTraj.h"
#include <algorithm>
#include <iterator>

namespace {
  template <typename FwdIterator, typename F>
  F for_each_adjacent_pair( FwdIterator begin, FwdIterator end, F f ) {
    using arg_t = decltype( *begin );
    using std::next;
    if ( begin != end )
      std::mismatch( next( begin ), end, begin,
                     // note the reversal of order here!
                     // first range is next(begin),end,
                     // second range is begin,prev(end)
                     [&]( arg_t rhs, arg_t lhs ) {
                       f( lhs, rhs );
                       return true;
                     } );
    return f;
  }
} // namespace

IMaterialLocator::Intersections MaterialLocatorBase::intersect_point( const Gaudi::XYZPoint&  p,
                                                                      const Gaudi::XYZVector& v, std::any& accelCache,
                                                                      IGeometryInfo const& geometry ) const {
  ::Intersections                 origintersepts = intersect_volumes( p, v, accelCache, geometry );
  const auto                      dz             = v.z();
  const auto                      z1             = p.z();
  const auto                      tx             = v.x() / v.z();
  const auto                      ty             = v.y() / v.z();
  IMaterialLocator::Intersections intersepts;
  intersepts.reserve( origintersepts.size() );
  std::transform( origintersepts.begin(), origintersepts.end(), std::back_inserter( intersepts ),
                  [&]( auto const& i ) -> IMaterialLocator::Intersection {
                    return { z1 + dz * i.first.first, z1 + dz * i.first.second, tx, ty, i.second };
                  } );
  return intersepts;
}

inline double pointerror( const LHCb::StateVector& begin, const LHCb::StateVector& end, const LHCb::StateVector& mid ) {
  double rc( 0 );
  auto   dz = end.z() - begin.z();
  if ( fabs( dz ) > TrackParameters::propagationTolerance ) {
    const auto tx = ( end.x() - begin.x() ) / dz;
    const auto ty = ( end.y() - begin.y() ) / dz;
    dz            = mid.z() - begin.z();
    const auto dx = begin.x() + dz * tx - mid.x();
    const auto dy = begin.y() + dz * ty - mid.y();
    rc            = std::sqrt( ( dx * dx ) + ( dy * dy ) );
  }
  return rc;
}

inline double linearerror( const LHCb::StateVector& origin, const LHCb::StateVector& destination ) {
  // calculate deviation from a parabola
  const auto dz = destination.z() - origin.z();
  const auto dx = origin.x() + origin.tx() * dz - destination.x();
  const auto dy = origin.y() + origin.ty() * dz - destination.y();
  return 0.25 * std::sqrt( ( dx * dx ) + ( dy * dy ) );
}

IMaterialLocator::Intersections MaterialLocatorBase::intersect( const LHCb::ZTrajectory<double>& traj,
                                                                std::any&                        accelCache,
                                                                IGeometryInfo const&             geometry ) const {
  IMaterialLocator::Intersections intersepts;
  if ( std::abs( traj.endRange() - traj.beginRange() ) > TrackParameters::propagationTolerance ) {
    // The material locators can only use straight lines, so we
    // approximate the trajectory by straight lines. The less
    // intervals the better. We determine the number of intervals from
    // the maximum allowed deviation. Because 'queuering' the
    // trajectory for a state is potentially expensive (e.g. for the
    // tracktraj), we also limit the number of calls to the trajectory
    // as much as possible. There are two simple ways of calculating
    // the error: Either we can queuery the trajectory halfway
    // ('midpointerror'), or we can estimate the halfway deviation
    // from extrapolating to the end ('linearerror'). The latter is
    // cheaper and more conservative; the former is more optimal
    // if tracks aren't very quadratic.

    // The first two ndoes are just the endpoints. We sort the in z to
    // make things easier.
    const size_t                 maxnumnodes = m_maxNumIntervals + 1;
    std::list<LHCb::StateVector> nodes       = { traj.stateVector( std::min( traj.beginRange(), traj.endRange() ) ),
                                                 traj.stateVector( std::max( traj.beginRange(), traj.endRange() ) ) };
    auto                         inode       = nodes.begin();

    // reference states for this trajectory. may be empty.
    if ( m_maxDeviationAtRefstates > 0 ) {
      const auto refstates = traj.refStateVectors();
      // First insert nodes at the reference positions, if required
      if ( !refstates.empty() ) {
        std::list<LHCb::StateVector>::iterator nextnode;
        while ( ( nextnode = next( inode ) ) != nodes.end() && nodes.size() < maxnumnodes ) {
          auto   worstref = refstates.end();
          double reldeviation( 0 );
          for ( auto iref = refstates.begin(); iref != refstates.end(); ++iref )
            if ( inode->z() < iref->z() && iref->z() < nextnode->z() ) {
              double thisdeviation = pointerror( *inode, *nextnode, *iref );
              double thisreldeviation =
                  thisdeviation /
                  ( iref->z() < StateParameters::ZEndVelo ? m_maxDeviationAtVeloRefstates : m_maxDeviationAtRefstates );
              if ( thisreldeviation > reldeviation ) {
                reldeviation = thisreldeviation;
                worstref     = iref;
              }
            }
          if ( reldeviation > 1 && worstref != refstates.end() ) {
            nodes.insert( nextnode, *worstref );
          } else
            ++inode;
        }
      }
    }

    // now the usual procedure
    inode = nodes.begin();
    double                                 worstdeviation( 0 );
    auto                                   worstnode = inode;
    std::list<LHCb::StateVector>::iterator nextnode;
    while ( ( nextnode = next( inode ) ) != nodes.end() && nodes.size() < maxnumnodes ) {
      // make sure we are fine at the midpoint
      auto       midpoint  = traj.stateVector( 0.5 * ( inode->z() + nextnode->z() ) );
      const auto deviation = pointerror( *inode, *nextnode, midpoint );
      if ( deviation > m_maxDeviation ) {
        nodes.insert( nextnode, midpoint );
      } else {
        if ( deviation > worstdeviation ) {
          worstdeviation = deviation;
          worstnode      = inode;
        }
        ++inode;
      }
    }

    // issue a warning if we didn't make it
    if ( nodes.size() == maxnumnodes )
      Warning( "Trajectory approximation did not reach desired accuracy. ", StatusCode::SUCCESS, 0 ).ignore();

    // debug output
    if ( msgLevel( MSG::VERBOSE ) || ( msgLevel( MSG::DEBUG ) && nodes.size() == maxnumnodes ) ) {
      debug() << "Trajectory approximation: numnodes=" << nodes.size() << ", deviation=" << worstdeviation
              << " at z= " << 0.5 * ( worstnode->z() + next( worstnode )->z() ) << endmsg;
      if ( msgLevel( MSG::DEBUG ) )
        for_each_adjacent_pair( nodes.begin(), nodes.end(),
                                [&]( const LHCb::StateVector& l, const LHCb::StateVector& r ) {
                                  auto midpoint = traj.stateVector( 0.5 * ( l.z() + r.z() ) );
                                  debug() << "interval: "
                                          << "(" << l.z() << ", " << r.z() << ")"
                                          << " ---> midpoint deviation: " << pointerror( l, r, midpoint ) << endmsg;
                                } );
    }

    // Now create intersections for each of the intervals.
    auto p1 = nodes.front().position();
    for ( inode = nodes.begin(); ( nextnode = std::next( inode ) ) != nodes.end(); ++inode ) {
      auto p2 = nextnode->position();
      try {
        IMaterialLocator::Intersections tmpintersepts =
            MaterialLocatorBase::intersect_point( p1, p2 - p1, accelCache, geometry );
        intersepts.insert( intersepts.end(), tmpintersepts.begin(), tmpintersepts.end() );
        p1 = p2;
      } catch ( GaudiException& exception ) {
        error() << "propagating pos1, pos2: " << p1 << " " << p2 << " " << traj.beginPoint() << " " << traj.endPoint()
                << endmsg;
        throw exception;
      }
    }
  }

  return intersepts;
}

// FIXME: add createCache() function + `std::any` argument here for the two buffers we use

void MaterialLocatorBase::applyMaterialCorrections( LHCb::State&                           stateAtTarget,
                                                    const IMaterialLocator::Intersections& intersepts, double zorigin,
                                                    const LHCb::Tr::PID pid, bool applyScatteringCorrection,
                                                    bool applyEnergyLossCorrection ) const {
  double ztarget  = stateAtTarget.z();
  bool   upstream = zorigin > ztarget;
  double qop      = stateAtTarget.qOverP();
  double pmass    = pid.mass();
  // loop over all intersections and do the work. note how we
  // explicitely keep the momentum constant. note that the way we
  // write this down, we rely on the fact that it is totally
  // irrelevant how the intersepts are sorted (because the propagation
  // is assumed to be linear.)

  // the only thing that is tricky is dealing with the fact that z1
  // and z2 need not be in increasing value, nor intersept.z1 and
  // intersept.z2. that makes calculating the overlap ('thickness') a
  // bit difficult. that's why we just reorder them.
  double zmin( zorigin ), zmax( ztarget );
  if ( upstream ) std::swap( zmin, zmax );
  const IStateCorrectionTool* dedxtool = ( pid.isElectron() ? &( *m_elecdedxtool ) : &( *m_dedxtool ) );

  // FIXME not every applyMaterialCorrections call needs a new state, right?
  // However changing yet another interface and passing three buffers seems somewhat unreasonable.
  // Better to rethink the whole structure, and leave this fix for now to avoid race conditions.
  auto ScatterToolBuffer = m_scatteringTool->createBuffer();
  auto DedxToolBuffer    = m_dedxtool->createBuffer();

  // Gaudi::TrackMatrix F = ROOT::Math::SMatrixIdentity();
  for ( auto isept : intersepts ) {
    const auto z1        = std::max( zmin, std::min( isept.z1, isept.z2 ) );
    const auto z2        = std::min( zmax, std::max( isept.z1, isept.z2 ) );
    const auto thickness = z2 - z1; // negative thickness means no overlap
    if ( thickness > TrackParameters::propagationTolerance ) {
      // double thickness = z2 - z1 ; // Why this? Was something else intended?

      // create a state. probably it is faster not to create it. but then we need to reset the noise every time.
      LHCb::State state;
      state.setQOverP( qop );
      state.setTx( isept.tx );
      state.setTy( isept.ty );

      // now add the wall
      if ( applyScatteringCorrection ) {
        m_scatteringTool->correctState( state, isept.material, ScatterToolBuffer, thickness, upstream, pmass );
      }
      if ( applyEnergyLossCorrection ) {
        dedxtool->correctState( state, isept.material, DedxToolBuffer, thickness, upstream, pmass );
      }

      // add the change in qOverP
      stateAtTarget.setQOverP( stateAtTarget.qOverP() + state.qOverP() - qop );

      // propagate the noise to the target. linear propagation, only
      // non-zero contributions
      const auto dz = ( upstream ? ztarget - z1 : ztarget - z2 );
      state.covariance()( 0, 0 ) += 2 * dz * state.covariance()( 2, 0 ) + dz * dz * state.covariance()( 2, 2 );
      state.covariance()( 2, 0 ) += dz * state.covariance()( 2, 2 );
      state.covariance()( 1, 1 ) += 2 * dz * state.covariance()( 3, 1 ) + dz * dz * state.covariance()( 3, 3 );
      state.covariance()( 3, 1 ) += dz * state.covariance()( 3, 3 );
      stateAtTarget.covariance() += state.covariance();
    }
  }
}
