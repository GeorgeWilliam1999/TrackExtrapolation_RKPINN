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

#include "Event/StateVector.h"

#include "Core/FloatComparison.h"
#include "DetDesc/DetectorElement.h"
#include "DetDesc/GenericConditionAccessorHolder.h"
#include "TrackFieldExtrapolatorBase.h"
#include "TrackInterfaces/ITrackExtrapolator.h"

#include "GaudiKernel/PhysicalConstants.h"
#include "GaudiKernel/ToolHandle.h"
#include "LHCbAlgs/Consumer.h"

#include "boost/format.hpp"

class ExtrapolatorTester : public LHCb::Algorithm::Consumer<void( DetectorElement const& ),
                                                            LHCb::Algorithm::Traits::usesConditions<DetectorElement>> {

public:
  ExtrapolatorTester( const std::string& name, ISvcLocator* pSvcLocator )
      : Consumer( name, pSvcLocator, { KeyValue{ "StandardGeometryTop", LHCb::standard_geometry_top } } ) {}

  void operator()( DetectorElement const& ) const override;

private:
  ToolHandleArray<ITrackExtrapolator> m_extraps{ this, "Extrapolators", {} };
};

DECLARE_COMPONENT( ExtrapolatorTester )

void ExtrapolatorTester::operator()( DetectorElement const& lhcb ) const {

  std::string prefix   = "Propagating ";
  auto        toStream = []( auto& os, LHCb::StateVector const& s ) -> decltype( auto ) {
    return os << boost::format( "( %7.3f, %7.3f, %5.4f, %5.4f )" ) % s.x() % s.y() % s.tx() % s.ty();
  };

  auto name = []( const auto& ex ) {
    std::string_view v = ex.name();
    v.remove_prefix( v.find( '.' ) + 1 );
    return v;
  };
  auto len = std::accumulate( m_extraps.begin(), m_extraps.end(), prefix.size(),
                              [&]( auto i, const auto& e ) { return std::max( i, name( *e ).size() ); } );
  if ( len > prefix.size() ) prefix.append( len - prefix.size(), ' ' );
  auto fmtName = [len, name]( auto& ex ) {
    auto n = name( ex );
    return std::string( len - n.size(), ' ' ).append( n );
  };

  const double z1 = 3000.;
  const double z2 = 7000.;

  const double qopmax = +0.0004; // 2.5 GeV
  const double qopmin = -0.0004;
  const double txmax  = +0.3;
  const double txmin  = -0.3;
  const double tymax  = +0.25;
  const double tymin  = -0.25;
  const int    nbins  = 11;

  const double dqop = ( qopmax - qopmin ) / ( nbins - 1 );
  const double dtx  = ( txmax - txmin ) / ( nbins - 1 );
  const double dty  = ( tymax - tymin ) / ( nbins - 1 );

  for ( int iqop = 0; iqop < nbins; ++iqop ) // grid in tx, ty, qop
    for ( int itx = 0; itx < nbins; ++itx )
      for ( int ity = 0; ity < nbins; ++ity ) {
        const double      qop = qopmin + iqop * dqop;
        const double      tx  = txmin + itx * dtx;
        const double      ty  = tymin + ity * dty;
        LHCb::StateVector origin;
        origin.setX( tx * z1 ); // assume tracks from (0,0,0)
        origin.setY( ty * z1 ); // assume tracks from (0,0,0)
        origin.setTx( tx );
        origin.setTy( ty );

        origin.setQOverP( qop );
        origin.setZ( z1 );

        toStream( always() << prefix << "    ", origin )
            << " with q*p = "
            << ( !LHCb::essentiallyZero( qop ) ? 1. / ( Gaudi::Units::GeV * qop )
                                               : std::numeric_limits<double>::infinity() )
            << " GeV from z = " << z1 << " to " << z2 << endmsg;
        for ( const auto& extrap : m_extraps ) {
          LHCb::StateVector  target = origin;
          Gaudi::TrackMatrix jacobian;
          extrap->propagate( target, z2, *lhcb.geometry(), &jacobian )
              .andThen( [&] { toStream( always() << fmtName( *extrap ) << " -> ", target ) << endmsg; } )
              .ignore();
        }
      }
}
