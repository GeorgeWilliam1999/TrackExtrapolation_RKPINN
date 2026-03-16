/*****************************************************************************\
* (c) Copyright 2000-2019 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "Gaudi/Timers.h"
#include "GaudiAlg/GaudiTupleAlg.h"
#include "GaudiKernel/PhysicalConstants.h"
#include "GaudiKernel/ToolHandle.h"

#include "Event/StateVector.h"
#include "TrackExtrapolator.h"
#include "TrackFieldExtrapolatorBase.h"
#include "TrackInterfaces/ITrackExtrapolator.h"

#include "DetDesc/DetectorElement.h"
#include "DetDesc/GenericConditionAccessorHolder.h"
#include "Event/State.h"
#include "GaudiKernel/SystemOfUnits.h"
#include "LHCbAlgs/Consumer.h"

#ifdef __x86_64__
using NanoTimer =
    Gaudi::Timers::GenericTimer<Gaudi::Timers::RdtscClock<std::chrono::nanoseconds>, std::chrono::nanoseconds>;
#else
using NanoTimer = Gaudi::Timers::GenericTimer<std::chrono::high_resolution_clock, std::chrono::nanoseconds>;
#endif

namespace LHCb {

  /** @class TrackExtrapolatorTesterSOA
   *  Benchmark algorithm that generates synthetic test states on a grid
   *  and times each configured extrapolator variant, writing per-track
   *  NTuples with accuracy and sub-operation timing breakdowns.
   */
  class TrackExtrapolatorTesterSOA
      : public Algorithm::Consumer<void( DetectorElement const& ),
                                   LHCb::Algorithm::Traits::usesBaseAndConditions<GaudiTupleAlg, DetectorElement>> {

  public:
    TrackExtrapolatorTesterSOA( const std::string& name, ISvcLocator* pSvcLocator )
        : Consumer( name, pSvcLocator, { { "StandardGeometryTop", standard_geometry_top } } ) {}

    void operator()( DetectorElement const& ) const override;

  private:
    ToolHandle<ITrackExtrapolator>      m_refextrap{ this, "ReferenceExtrapolator", "TrackSTEPExtrapolator" };
    Gaudi::Property<double>             m_zi{ this, "InitialZ", 3000. };
    Gaudi::Property<double>             m_zf{ this, "FinalZ", 7000. };
    Gaudi::Property<int>                m_nbins{ this, "NBins", 11 };
    ToolHandleArray<ITrackExtrapolator> m_extraps{ this, "Extrapolators", {} };
  };

  DECLARE_COMPONENT_WITH_ID( TrackExtrapolatorTesterSOA, "TrackExtrapolatorTesterSOA" )

} // namespace LHCb

void LHCb::TrackExtrapolatorTesterSOA::operator()( DetectorElement const& lhcb ) const {
  m_refextrap.retrieve().ignore();

  auto& geometry = *lhcb.geometry();

  // Create tuples for each extrapolator
  std::vector<std::pair<Tuple, ToolHandle<ITrackExtrapolator>>> extraps;
  for ( const auto& extrap : m_extraps ) {
    extraps.emplace_back( std::make_pair( nTuple( extrap.name(), "", CLID_ColumnWiseTuple ), extrap ) );
  }

  const double z1    = m_zi.value();
  const double z2    = m_zf.value();
  const int    nbins = m_nbins.value();

  // Generate synthetic test states on a grid (same range as ExtrapolatorTester)
  const double qopmax = +0.0004; // ~2.5 GeV
  const double qopmin = -0.0004;
  const double txmax  = +0.3;
  const double txmin  = -0.3;
  const double tymax  = +0.25;
  const double tymin  = -0.25;

  const double dqop = ( qopmax - qopmin ) / ( nbins - 1 );
  const double dtx  = ( txmax - txmin ) / ( nbins - 1 );
  const double dty  = ( tymax - tymin ) / ( nbins - 1 );

  for ( int iqop = 0; iqop < nbins; ++iqop )
    for ( int itx = 0; itx < nbins; ++itx )
      for ( int ity = 0; ity < nbins; ++ity ) {
        const double qop = qopmin + iqop * dqop;
        const double tx  = txmin + itx * dtx;
        const double ty  = tymin + ity * dty;

        // Create initial state with diagonal covariance
        Gaudi::TrackVector    trackVector{ tx * z1, ty * z1, tx, ty, qop };
        Gaudi::TrackSymMatrix cov = Gaudi::TrackSymMatrix();
        cov( 0, 0 )              = 1.0;
        cov( 1, 1 )              = 1.0;
        cov( 2, 2 )              = 1e-4;
        cov( 3, 3 )              = 1e-4;
        cov( 4, 4 )              = 1e-8;
        LHCb::State origin{ trackVector, cov, z1, LHCb::State::Location::FirstMeasurement };

        // Extrapolate with reference
        LHCb::State        reftarget = origin;
        Gaudi::TrackMatrix refjacobian;
        m_refextrap->propagate( reftarget, z2, &refjacobian, geometry ).ignore();

        // Extrapolate with each variant and record results
        for ( auto& extrap : extraps ) {
          LHCb::State        target = origin;
          Gaudi::TrackMatrix jacobian;

          const auto& extrapolator = extrap.second;
          StatusCode  sc;
          NanoTimer   timer;
          {
            auto timeit = timer();
            sc          = extrapolator->propagate( target, z2, &jacobian, geometry ).ignore();
          }
          auto time_ns = int( timer.stats().mean().count() );

          auto& roottuple = extrap.first;
          roottuple->column( "qop", origin.qOverP() ).ignore();
          roottuple->column( "x_i", origin.x() ).ignore();
          roottuple->column( "y_i", origin.y() ).ignore();
          roottuple->column( "tx_i", origin.tx() ).ignore();
          roottuple->column( "ty_i", origin.ty() ).ignore();

          roottuple->column( "x_f", reftarget.x() ).ignore();
          roottuple->column( "y_f", reftarget.y() ).ignore();
          roottuple->column( "tx_f", reftarget.tx() ).ignore();
          roottuple->column( "ty_f", reftarget.ty() ).ignore();

          roottuple->column( "dx", target.x() - reftarget.x() ).ignore();
          roottuple->column( "dy", target.y() - reftarget.y() ).ignore();
          roottuple->column( "dtx", target.tx() - reftarget.tx() ).ignore();
          roottuple->column( "dty", target.ty() - reftarget.ty() ).ignore();
          roottuple->column( "time", time_ns ).ignore();
          roottuple->column( "success", int( sc.getCode() ) ).ignore();

          // Sub-operation timing breakdown (if enabled on this extrapolator)
          auto* fieldExtrap = dynamic_cast<const TrackFieldExtrapolatorBase*>( &*extrapolator );
          if ( fieldExtrap && fieldExtrap->subTimersEnabled() ) {
            const auto& st = fieldExtrap->lastSubTimers();
            roottuple->column( "field_cycles", int( st.field_cycles ) ).ignore();
            roottuple->column( "deriv_cycles", int( st.deriv_cycles ) ).ignore();
            roottuple->column( "butcher_cycles", int( st.butcher_cycles ) ).ignore();
            roottuple->column( "jacobian_cycles", int( st.jacobian_cycles ) ).ignore();
            roottuple->column( "stepsize_cycles", int( st.stepsize_cycles ) ).ignore();
            roottuple->column( "total_cycles", int( st.total_cycles ) ).ignore();
            roottuple->column( "nsteps", st.nsteps ).ignore();
            roottuple->column( "nrejected", st.nrejected ).ignore();
          }

          roottuple->write().ignore();
        }
      }
}
