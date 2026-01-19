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
#include "Event/States.h"
#include "Event/Track.h"
#include "Event/Track_v3.h"
#include "GaudiKernel/SystemOfUnits.h"
#include "LHCbAlgs/Consumer.h"
#include "TrackKernel/TrackFunctors.h"
#include <map>
#include <string>

using v3_Tracks = LHCb::Event::v3::Tracks;
using SL        = LHCb::Event::v3::detail::StateLocation;
#ifdef __x86_64__
using NanoTimer =
    Gaudi::Timers::GenericTimer<Gaudi::Timers::RdtscClock<std::chrono::nanoseconds>, std::chrono::nanoseconds>;
#else
using NanoTimer = Gaudi::Timers::GenericTimer<std::chrono::high_resolution_clock, std::chrono::nanoseconds>;
#endif

namespace LHCb {

  class TrackExtrapolatorTesterSOA
      : public Algorithm::Consumer<void( v3_Tracks const&, DetectorElement const& ),
                                   LHCb::Algorithm::Traits::usesBaseAndConditions<GaudiTupleAlg, DetectorElement>> {

  public:
    TrackExtrapolatorTesterSOA( const std::string& name, ISvcLocator* pSvcLocator )
        : Consumer(
              name, pSvcLocator,
              { { "TrackContainer", TrackLocation::Default }, { "StandardGeometryTop", standard_geometry_top } } ) {}

    void fill_states_collection( Event::v3::States&, v3_Tracks const& ) const;
    void operator()( v3_Tracks const&, DetectorElement const& ) const override;

  private:
    ToolHandle<ITrackExtrapolator>      m_refextrap{ this, "ReferenceExtrapolator", "TrackSTEPExtrapolator" };
    Gaudi::Property<double>             m_zi{ this, "InitialZ", 0. };
    Gaudi::Property<double>             m_zf{ this, "FinalZ", 0. };
    ToolHandleArray<ITrackExtrapolator> m_extraps{ this, "Extrapolators", {} };
  };

  DECLARE_COMPONENT_WITH_ID( TrackExtrapolatorTesterSOA, "TrackExtrapolatorTesterSOA" )

} // namespace LHCb

void LHCb::TrackExtrapolatorTesterSOA::fill_states_collection( LHCb::Event::v3::States& states,
                                                               v3_Tracks const&         tracks ) const {

  states.reserve( tracks.size() );

  const auto init_SL = SL::FirstMeasurement;
  for ( auto const& track : tracks.scalar() ) {
    auto newState = states.emplace_back<SIMDWrapper::InstructionSet::Scalar>();
    auto ts       = track.state( init_SL );

    // assuming all input states are valid
    StatusCode sc{ StatusCode::SUCCESS };
    newState.field<LHCb::Event::v3::StatesTag::Success>().set( int( sc.getCode() ) );

    newState.field<LHCb::Event::v3::StatesTag::State>().setPosition( ts.x().cast(), ts.y().cast(), ts.z().cast() );
    newState.field<LHCb::Event::v3::StatesTag::State>().setDirection( ts.tx().cast(), ts.ty().cast() );
    newState.field<LHCb::Event::v3::StatesTag::State>().setQOverP( ts.qOverP().cast() );

    auto cov           = track.covariance( init_SL );
    auto x_x           = cov( 0, 0 );
    auto x_y           = cov( 0, 1 );
    auto x_tx          = cov( 0, 2 );
    auto x_ty          = cov( 0, 3 );
    auto x_qOverP      = cov( 0, 4 );
    auto y_y           = cov( 1, 1 );
    auto y_tx          = cov( 1, 2 );
    auto y_ty          = cov( 1, 3 );
    auto y_qOverP      = cov( 1, 4 );
    auto tx_tx         = cov( 2, 2 );
    auto tx_ty         = cov( 2, 3 );
    auto tx_qOverP     = cov( 2, 4 );
    auto ty_ty         = cov( 3, 3 );
    auto ty_qOverP     = cov( 3, 4 );
    auto qOverP_qOverP = cov( 4, 4 );

    newState.field<LHCb::Event::v3::StatesTag::Covariance>().set( x_x, x_y, x_tx, x_ty, x_qOverP, y_y, y_tx, y_ty,
                                                                  y_qOverP, tx_tx, tx_ty, tx_qOverP, ty_ty, ty_qOverP,
                                                                  qOverP_qOverP );

  } // states collection created
}

void LHCb::TrackExtrapolatorTesterSOA::operator()( v3_Tracks const& tracks, DetectorElement const& lhcb ) const {
  m_refextrap.retrieve().ignore();

  auto& geometry = *lhcb.geometry();

  std::vector<std::pair<Tuple, ToolHandle<ITrackExtrapolator>>> extraps;
  std::vector<LHCb::Event::v3::States>                          states_collections;
  std::vector<LHCb::Event::v3::States>                          out_states_collections;

  LHCb::Event::v3::States ref_states{};
  fill_states_collection( ref_states, tracks ); // create collection of states

  for ( const auto& extrap : m_extraps ) {
    LHCb::Event::v3::States states{};
    fill_states_collection( states, tracks );
    extraps.emplace_back( std::make_pair( nTuple( extrap.name(), "", CLID_ColumnWiseTuple ), extrap ) );
    states_collections.emplace_back( std::move( states ) );
  }

  const double z2 = m_zf.value();

  // propagate states collection(s)
  auto out_ref_states = m_refextrap->propagate( ref_states, z2, geometry );
  int  counter        = 0;
  for ( auto& pair : extraps ) {
    const auto& extrapolator = pair.second;
    const auto& states       = states_collections[counter];
    auto        out_states   = extrapolator->propagate( states, z2, geometry );
    out_states_collections.emplace_back( std::move( out_states ) );
    counter++;
  }

  // do same thing but with LHCb::State
  int        track_index = 0;
  const auto init_SL     = SL::FirstMeasurement;

  for ( auto const& track : tracks.scalar() ) {

    auto state = track.state( init_SL );

    Gaudi::TrackVector    trackVector{ state.x().cast(), state.y().cast(), state.tx().cast(), state.ty().cast(),
                                    state.qOverP().cast() };
    Gaudi::TrackSymMatrix cov      = Gaudi::TrackSymMatrix();
    auto                  simd_cov = track.covariance( init_SL );
    for ( int i = 0; i < 5; i++ )
      for ( int j = 0; j < 5; j++ ) cov( i, j ) = simd_cov( i, j ).cast();
    LHCb::State origin{ trackVector, cov, state.z().cast(), LHCb::State::Location::FirstMeasurement };

    // extrapolate the reference
    LHCb::State        reftarget = origin;
    Gaudi::TrackMatrix refjacobian;
    m_refextrap->propagate( reftarget, z2, &refjacobian, geometry ).ignore();

    // now do the same for the others
    counter = 0;
    for ( auto& extrap : extraps ) {

      LHCb::State        target = origin;
      Gaudi::TrackMatrix jacobian, dummyjacobian;

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

      // fill differences between SoA and non-SoA versions
      auto& out_states    = out_states_collections[counter];
      auto  current_state = out_states.scalar()[track_index];
      auto  soa_state     = current_state.get<LHCb::Event::v3::StatesTag::State>();
      auto  soa_cov       = current_state.get<LHCb::Event::v3::StatesTag::Covariance>();
      auto  soa_success   = current_state.get<LHCb::Event::v3::StatesTag::Success>().cast();

      roottuple->column( "soa_success", int( soa_success ) ).ignore();

      roottuple->column( "soa_dx", target.x() - soa_state.x().cast() ).ignore();
      roottuple->column( "soa_dy", target.y() - soa_state.y().cast() ).ignore();
      roottuple->column( "soa_dtx", target.tx() - soa_state.tx().cast() ).ignore();
      roottuple->column( "soa_dty", target.ty() - soa_state.ty().cast() ).ignore();

      auto cov = target.covariance();
      roottuple->column( "soa_d_xx", cov( 0, 0 ) - soa_cov.x_x().cast() ).ignore();
      roottuple->column( "soa_d_xy", cov( 0, 1 ) - soa_cov.x_y().cast() ).ignore();
      roottuple->column( "soa_d_xtx", cov( 0, 2 ) - soa_cov.x_tx().cast() ).ignore();
      roottuple->column( "soa_d_xty", cov( 0, 3 ) - soa_cov.x_ty().cast() ).ignore();
      roottuple->column( "soa_d_xQoverP", cov( 0, 4 ) - soa_cov.x_QoverP().cast() ).ignore();
      roottuple->column( "soa_d_yy", cov( 1, 1 ) - soa_cov.y_y().cast() ).ignore();
      roottuple->column( "soa_d_ytx", cov( 1, 2 ) - soa_cov.y_tx().cast() ).ignore();
      roottuple->column( "soa_d_yty", cov( 1, 3 ) - soa_cov.y_ty().cast() ).ignore();
      roottuple->column( "soa_d_yQoverP", cov( 1, 4 ) - soa_cov.y_QoverP().cast() ).ignore();
      roottuple->column( "soa_d_txtx", cov( 2, 2 ) - soa_cov.tx_tx().cast() ).ignore();
      roottuple->column( "soa_d_txty", cov( 2, 3 ) - soa_cov.tx_ty().cast() ).ignore();
      roottuple->column( "soa_d_txQoverP", cov( 2, 4 ) - soa_cov.tx_QoverP().cast() ).ignore();
      roottuple->column( "soa_d_tyty", cov( 3, 3 ) - soa_cov.ty_ty().cast() ).ignore();
      roottuple->column( "soa_d_tyQoverP", cov( 3, 4 ) - soa_cov.ty_QoverP().cast() ).ignore();
      roottuple->column( "soa_d_QoverPQoverP", cov( 4, 4 ) - soa_cov.QoverP_QoverP().cast() ).ignore();

      roottuple->column( "xx", cov( 0, 0 ) ).ignore();
      roottuple->column( "xy", cov( 0, 1 ) ).ignore();
      roottuple->column( "xtx", cov( 0, 2 ) ).ignore();
      roottuple->column( "xty", cov( 0, 3 ) ).ignore();
      roottuple->column( "xQoverP", cov( 0, 4 ) ).ignore();
      roottuple->column( "yy", cov( 1, 1 ) ).ignore();
      roottuple->column( "ytx", cov( 1, 2 ) ).ignore();
      roottuple->column( "yty", cov( 1, 3 ) ).ignore();
      roottuple->column( "yQoverP", cov( 1, 4 ) ).ignore();
      roottuple->column( "txtx", cov( 2, 2 ) ).ignore();
      roottuple->column( "txty", cov( 2, 3 ) ).ignore();
      roottuple->column( "txQoverP", cov( 2, 4 ) ).ignore();
      roottuple->column( "tyty", cov( 3, 3 ) ).ignore();
      roottuple->column( "tyQoverP", cov( 3, 4 ) ).ignore();
      roottuple->column( "QoverPQoverP", cov( 4, 4 ) ).ignore();

      roottuple->write().ignore();
      counter++;
    }
    track_index++;
  }
}
