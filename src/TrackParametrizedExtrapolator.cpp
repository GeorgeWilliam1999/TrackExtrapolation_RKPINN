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
#include "BasisFunctions.h"
#include "TrackExtrapolator.h"

#include "DetDesc/DetectorElement.h"
#include "DetDesc/GenericConditionAccessorHolder.h"
#include "DetDesc/ValidDataObject.h"
#include "Event/StateParameters.h"
#include "Event/TrackParameters.h"
#include "GaudiKernel/IUpdateManagerSvc.h"
#include "Kernel/ILHCbMagnetSvc.h"

#include <Eigen/Dense>

namespace {

  using namespace BasisFunctions;

  struct Plane {
    Plane() = default;
    Plane( double z, double x1, double x2, double y1, double y2 )
        : z( z ), xmin( x1 ), xmax( x2 ), ymin( y1 ), ymax( y2 ) {}
    Plane( double z, double txmax, double tymax )
        : z( z ), xmin( -z * txmax ), xmax( z * txmax ), ymin( -z * tymax ), ymax( z * tymax ) {}
    double z{ 0 };
    double xmin{ 0 };
    double xmax{ 0 };
    double ymin{ 0 };
    double ymax{ 0 };
  };

  template <class FunctionBasisT>
  class PropagatorParametrization {
  public:
    typedef FunctionBasisT                                     FunctionBasis;
    typedef Eigen::Matrix<double, 4, FunctionBasis::NumValues> Coefficients;

  private:
    Coefficients          m_coefficients;
    std::array<double, 5> m_offset;
    std::array<double, 5> m_scale;
    const double          m_dz;

  public:
    PropagatorParametrization( double deltaZ ) : m_coefficients{ Coefficients::Zero() }, m_dz( deltaZ ) {
      m_offset.fill( 0.0 );
      m_scale.fill( 1.0 );
    }
    // change a state vector into arguments on (approximately) range [-1,1]
    auto arguments( const Gaudi::TrackVector& statein ) const {
      typename FunctionBasis::Arguments xprime;
      for ( int i = 0; i < 5; ++i ) { xprime[i] = m_scale[i] * ( statein[i] - m_offset[i] ); }
      return xprime;
    }
    // propagate a state vector.
    void propagate( Gaudi::TrackVector& statein, Gaudi::TrackMatrix* j = 0 ) const {
      const auto   xprime           = arguments( statein );
      const auto   basisfunctionval = FunctionBasis{}.evaluate( xprime );
      const auto   val              = m_coefficients * basisfunctionval;
      const double qop              = statein[4];
      statein[0] += m_dz * statein[2] + qop * val( 0 );
      statein[1] += m_dz * statein[3] + qop * val( 1 );
      statein[2] += qop * val( 2 );
      statein[3] += qop * val( 3 );
      if ( j ) {
        // compute the jacobian
        auto& m = *j;
        // let's do this the safe way, for now
        for ( int irow = 0; irow < 5; ++irow ) {
          for ( int icol = 0; icol < 4; ++icol ) m( irow, icol ) = 0;
          ;
          m( irow, irow ) = 1;
        }
        // add the trivial part
        m( 0, 2 ) += m_dz;
        m( 1, 3 ) += m_dz;
        // add the rest
        const auto                  basisderivatives = FunctionBasis{}.evaluateDerivative( xprime );
        Eigen::Matrix<double, 4, 5> derivatives      = m_coefficients * basisderivatives;
        for ( int irow = 0; irow < 4; ++irow ) {
          for ( int icol = 0; icol < 5; ++icol ) m( irow, icol ) += qop * m_scale[icol] * derivatives( irow, icol );
          // don't forget to add 2nd q/p contribution (the most
          // important one!
          m( irow, 4 ) += val( irow );
        }
      }
    } // end of propagate

    auto&  offset() { return m_offset; }
    auto&  scale() { return m_scale; }
    auto&  coefficients() { return m_coefficients; }
    double varmin( int ivar ) const { return m_offset[ivar] - 1.0 / m_scale[ivar]; }
    double varmax( int ivar ) const { return m_offset[ivar] + 1.0 / m_scale[ivar]; }
  };

  class InterPlaneParametrization {
  public:
    // enum{ OrderX = 0, OrderY = 0, OrderTx = 2, OrderTy = 2, OrderQoP= 4 } ;
    enum : std::size_t { OrderX = 0, OrderY = 0, OrderTx = 2, OrderTy = 2, OrderQoP = 4 };
    template <size_t N>
    using Poly1D = Polynomial1D<N>;
    typedef BasicFunctionProduct<
        Poly1D<OrderX>,
        BasicFunctionProduct<
            Poly1D<OrderY>,
            BasicFunctionProduct<Poly1D<OrderTx>, BasicFunctionProduct<Poly1D<OrderTy>, Poly1D<OrderQoP>>>>>
                                                     FunctionBasis;
    typedef PropagatorParametrization<FunctionBasis> Propagator;

  private:
    Plane m_plane1;
    Plane m_plane2;
    // the number of bins in x and y. these will become template parameters?
    size_t                  m_numbinsX;
    size_t                  m_numbinsY;
    std::vector<Propagator> m_propagators;

  public:
    size_t       numbinsX() const { return m_numbinsX; }
    size_t       numbinsY() const { return m_numbinsY; }
    const Plane& plane1() const { return m_plane1; }
    const Plane& plane2() const { return m_plane2; }
    auto&        propagators() { return m_propagators; }
    double       dz() const { return m_plane2.z - m_plane1.z; }

    InterPlaneParametrization( const Plane& p1, const Plane& p2 )
        : m_plane1{ p1 }
        , m_plane2{ p2 }
        , m_numbinsX{ 50 }
        , m_numbinsY{ 50 }
        , m_propagators( m_numbinsX * m_numbinsY, Propagator( p2.z - p1.z ) ) {
      // do some work for every bin
      const double dx = ( m_plane1.xmax - m_plane1.xmin ) / ( m_numbinsX );
      const double dy = ( m_plane1.ymax - m_plane1.ymin ) / ( m_numbinsY );

      for ( size_t xbin = 0; xbin < m_numbinsX; ++xbin )
        for ( size_t ybin = 0; ybin < m_numbinsY; ++ybin ) {
          const double xmin     = m_plane1.xmin + dx * xbin;
          const double xmax     = xmin + dx;
          const double ymin     = m_plane1.ymin + dy * ybin;
          const double ymax     = ymin + dy;
          const size_t thexybin = xbin + m_numbinsX * ybin;
          // offsets are set such that the mean of the argument becomes about zero
          auto& offset = m_propagators[thexybin].offset();
          auto& scale  = m_propagators[thexybin].scale();
          offset[0]    = 0.5 * ( xmax + xmin );
          offset[1]    = 0.5 * ( ymax + ymin );
          // these are substantially more tricky. this only works far
          // away from the ip and only when propagating to larger z.
          const double z = m_plane1.z;
          offset[2]      = z > 0 ? offset[0] / z : 0;
          offset[3]      = z > 0 ? offset[1] / z : 0;
          offset[4]      = 0;
          // scales are set such that the min/max of the argument become about -/+1
          scale[0] = 2.0 / ( xmax - xmin );
          scale[1] = 2.0 / ( ymax - ymin );
          scale[2] = 1. / 0.1; // 100 mrad ?! too little?
          scale[3] = 1. / 0.1; // 100 mrad ?!
          scale[4] = 2500.;    // 1/2.5GeV
        }
    }

    size_t xybin( const double x, const double y ) const {
      size_t xbin = ( x - m_plane1.xmin ) / ( m_plane1.xmax - m_plane1.xmin ) * m_numbinsX;
      size_t ybin = ( y - m_plane1.ymin ) / ( m_plane1.ymax - m_plane1.ymin ) * m_numbinsY;
      xbin        = ( xbin >= m_numbinsX ? m_numbinsX - 1 : xbin );
      ybin        = ( ybin >= m_numbinsY ? m_numbinsY - 1 : ybin );
      return xbin + m_numbinsX * ybin;
    }

    const auto& propagator( const double x, const double y ) const { return m_propagators[xybin( x, y )]; }

    void propagate( Gaudi::TrackVector& state, Gaudi::TrackMatrix* m ) const {
      size_t thexybin = xybin( state[0], state[1] );
      m_propagators[thexybin].propagate( state, m );
    }
  }; // end of InterPlaneParametrization

  // accumulator for training a propagator parametrization
  template <class PropagatorParametrization>
  class PropagatorAccumulator {
  private:
    typedef typename PropagatorParametrization::FunctionBasis FunctionBasis;
    enum { Dim = FunctionBasis::NumValues, NumFunctions = 4 };
    PropagatorParametrization*                              m_client;
    size_t                                                  m_numentries;
    std::array<Eigen::Matrix<double, Dim, 1>, NumFunctions> m_first;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>   m_second;

  public:
    PropagatorAccumulator( PropagatorParametrization& client ) : m_client{ &client }, m_numentries( 0 ) {
      m_second = Eigen::Matrix<double, Dim, Dim>::Zero();
      m_first.fill( Eigen::Matrix<double, Dim, 1>::Zero() );
    }

    // add one function value
    template <class Delta>
    void add( const typename FunctionBasis::Arguments& args, const Delta& value ) {
      auto functionvalues = FunctionBasis{}.evaluate( args );
      ++m_numentries;
      for ( int i = 0; i < NumFunctions; ++i ) m_first[i] += value[i] * functionvalues;
      m_second += functionvalues * functionvalues.transpose();
    }
    // compute coefficients, in 4 steps
    void update() {
      for ( int i = 0; i < NumFunctions; ++i ) {
        // compute
        Eigen::Matrix<double, Dim, 1> coefficients = m_second.ldlt().solve( m_first[i] );
        // now update the client
        for ( int j = 0; j < Dim; ++j ) { m_client->coefficients()( i, j ) = coefficients( j ); }
      }
    }
    // value is really the part after dividing by q/p
    template <class Delta>
    void add( const Gaudi::TrackVector& in, const Delta& value ) {
      auto args = m_client->arguments( in );
      add( args, value );
    }
    // reset all accumulators
    void reset() {
      m_numentries = 0;
      m_first.fill( Eigen::Matrix<double, Dim, 1>::Zero() );
      m_second = Eigen::Matrix<double, Dim, Dim>::Zero();
    }
  };
} // namespace

/**
 *  A TrackParametrizedExtrapolator is a TrackExtrapolator with access to the magnetic field
 *
 *  @author Wouter Hulsbergen
 *  @date   16/09/2016
 */
class TrackParametrizedExtrapolator : public LHCb::DetDesc::ConditionAccessorHolder<TrackExtrapolator> {

public:
  typedef Gaudi::XYZVector FieldVector;
  typedef Gaudi::Matrix3x3 FieldGradient;

  using ConditionAccessorHolder::ConditionAccessorHolder;
  StatusCode initialize() override;
  using TrackExtrapolator::propagate;

  /// the one function that we need to implement
  /// Propagate a state vector from zOld to zNew
  /// Transport matrix is calulated when transMat pointer is not NULL
  StatusCode propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew, Gaudi::TrackMatrix* transMat,
                        IGeometryInfo const& geometry, LHCb::Tr::PID pid = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

private:
  const ToolHandle<ITrackExtrapolator> m_refextrapolator{ this, "RefExtrapolator", "TrackRungeKuttaExtrapolator" };

  Gaudi::Property<std::string> m_standardGeometry_address{ this, "StandardGeometryTop", LHCb::standard_geometry_top };

  struct Parameters {
    std::vector<InterPlaneParametrization> forwardpars;
    std::vector<InterPlaneParametrization> backwardpars;
  };
  ConditionAccessor<Parameters> m_parameters{ this, name() + "-Parameters" };
  Parameters                    computeParameters( LHCb::Detector::DeLHCb const&, ITrackExtrapolator const& );
};

DECLARE_COMPONENT( TrackParametrizedExtrapolator )

StatusCode TrackParametrizedExtrapolator::initialize() {
  return ConditionAccessorHolder::initialize().andThen( [&] {
    addConditionDerivation(
        { m_standardGeometry_address }, m_parameters.key(),
        [&]( LHCb::Detector::DeLHCb const& lhcb ) { return computeParameters( lhcb, *m_refextrapolator ); } );
    return StatusCode::SUCCESS;
  } );
}

TrackParametrizedExtrapolator::Parameters
TrackParametrizedExtrapolator::computeParameters( LHCb::Detector::DeLHCb const& lhcb,
                                                  ITrackExtrapolator const&     refextrapolator ) {
  // do we do upstream and downstream simultaneously?
  // do we start every trajectory from about the origin?
  // do we do all bins simultaneously?
  std::vector<InterPlaneParametrization> forwardpars;
  std::vector<InterPlaneParametrization> backwardpars;
  forwardpars.reserve( 3 );
  backwardpars.reserve( 3 );

  // lets's create a few planes
  const double txmax = 0.3;
  const double tymax = 0.25;
  Plane        endvelo{ StateParameters::ZEndVelo, txmax, tymax };
  Plane        beginUT{ 2200., txmax, tymax }; // need to add ZBegUT to StateParemeters
  Plane        endUT{ StateParameters::ZEndUT, txmax, tymax };
  Plane        beginT{ StateParameters::ZBegT, txmax, tymax };
  forwardpars.push_back( InterPlaneParametrization( endvelo, beginUT ) ); // 770 to 2200
  forwardpars.push_back( InterPlaneParametrization( endvelo, beginT ) );  // 770 to 7500
  forwardpars.push_back( InterPlaneParametrization( endUT, beginT ) );    // 2700 to 7500
  for ( const auto& p : forwardpars ) backwardpars.push_back( InterPlaneParametrization( p.plane2(), p.plane1() ) );

  info() << "Start training." << endmsg;
  chronoSvc()->chronoStart( "Training" );
  // easier if we have all of them in one container
  std::vector<InterPlaneParametrization*> planepars;
  for ( auto& i : forwardpars ) planepars.push_back( &i );
  for ( auto& i : backwardpars ) planepars.push_back( &i );
  for ( auto& i : planepars ) {
    auto&        par        = *i;
    const size_t npoints[5] = { par.OrderX + 1, par.OrderY + 1, par.OrderTx + 1, par.OrderTy + 1, par.OrderQoP + 1 };
    for ( size_t xbin = 0; xbin < par.numbinsX(); ++xbin ) {
      for ( size_t ybin = 0; ybin < par.numbinsY(); ++ybin ) {
        auto&        binprop = par.propagators()[xbin + par.numbinsX() * ybin];
        const double dz      = par.dz();

        // create the accumulator used to compute the polynominal coefficients
        PropagatorAccumulator<InterPlaneParametrization::Propagator> accumulator( binprop );
        // compute the grid points. (we do this first, because it
        // takes a bit of time for the Chebychev nodes.)
        std::vector<double> grid[5];
        for ( size_t j = 0; j < 5; ++j ) {
          if ( npoints[j] == 1 )
            grid[j].push_back( 0.5 * ( binprop.varmax( j ) + binprop.varmin( j ) ) );
          else {
            for ( size_t i = 0; i < npoints[j]; ++i ) {
              double xi = cos( M_PI * ( i + 0.5 ) / double( npoints[j] ) );
              grid[j].push_back( 0.5 * ( ( 1 - xi ) * binprop.varmin( j ) + ( 1 + xi ) * binprop.varmax( j ) ) );
            } // for loop closed
          }   // else loop closed
        }     // for loop closed

        // run over a grid of values in x,y,tx,ty,qop.
        StatusCode sc = StatusCode::SUCCESS;

        // FIXME: if one of the propagations fails, we have too few
        // points to determine the parameters. so, we need a fall back
        // solution that allows to add extra points.
        for ( const auto& x : grid[0] )
          for ( const auto& y : grid[1] )
            for ( const auto& tx : grid[2] )
              for ( const auto& ty : grid[3] )
                for ( const auto& qop : grid[4] ) {
                  Gaudi::TrackVector statein{ x, y, tx, ty, qop };
                  // const double qop = statein[4] ;
                  Gaudi::TrackVector stateout = statein;
                  if ( std::abs( qop ) > 1e-9 ) {
                    sc = refextrapolator.propagate( stateout, par.plane1().z, par.plane2().z, *lhcb.geometry() );
                    // FIX ME: do we only accept propagation that actually
                    // end up in the target plane?
                    if ( sc.isSuccess() ) {
                      // compute the deviation from a straight line
                      Gaudi::TrackVector delta = stateout - statein;
                      delta[0] -= dz * statein[2];
                      delta[1] -= dz * statein[3];
                      // now divide by qOverP, which we can only do if it is non-zero
                      delta /= qop;
                      // finally, accumulate
                      accumulator.add( statein, delta );
                      // std::cout << "A:" <<  statein << " " << delta[0] << " " << stateout[0] << std::endl ;
                    } else {
                      info() << "propagation failed! " << statein << endmsg;
                    }
                  } else {
                    // if qop==0, use the jacobian instead. this only
                    // works if step size is small enough.  FIXME
                    Gaudi::TrackMatrix jacobian;
                    sc = refextrapolator.propagate( stateout, par.plane1().z, par.plane2().z, &jacobian,
                                                    *lhcb.geometry() );
                    if ( sc.isSuccess() ) {
                      std::array<double, 4> delta = { jacobian( 0, 4 ), jacobian( 1, 4 ), jacobian( 2, 4 ),
                                                      jacobian( 3, 4 ) };
                      accumulator.add( statein, delta );
                    } else {
                      info() << "propagation failed! " << statein << endmsg;
                    }
                  }
                }
        //
        accumulator.update();
      }
    }
  }
  chronoSvc()->chronoStop( "Training" );
  info() << "Ready training." << endmsg;

  return { forwardpars, backwardpars };
}

StatusCode TrackParametrizedExtrapolator::propagate( Gaudi::TrackVector& in, double z1, double z2,
                                                     Gaudi::TrackMatrix* m, IGeometryInfo const& geometry,
                                                     LHCb::Tr::PID /*pid*/,
                                                     const LHCb::Magnet::MagneticFieldGrid* ) const {
  // FIXME getting back this derived condition at each call of propagate will be
  // extremely slow. This should be cached by the caller and passed to propagate.
  // Now this would mean a change of API in all propagate methods of ITrackExtrapolator
  // so a major code change. And on the other side this TrackParametrizedExtrapolator
  // is not used at all at the moment. So let's keep it like this for the moment
  auto const& params = m_parameters.get();
  // new approach: we take the interplane-extrapolator that has the
  // largest overlap in z. for now we only use it if it is fully
  // contained.
  StatusCode                       sc        = StatusCode::SUCCESS;
  int                              direction = z1 < z2 ? +1 : -1;
  auto&                            container = direction == +1 ? params.forwardpars : params.backwardpars;
  const InterPlaneParametrization* thepar( 0 );
  for ( auto& par : container ) {
    // check that it is contained
    if ( ( ( par.plane1().z - z1 ) * direction + TrackParameters::propagationTolerance ) >= 0 &&
         ( ( z2 - par.plane2().z ) * direction + TrackParameters::propagationTolerance >= 0 ) ) {
      if ( thepar == 0 || std::abs( thepar->dz() ) < std::abs( par.dz() ) ) { thepar = &par; }
    }
  }

  // the logic is such that we don't multiply jacobians if we are not
  // using the plane approximation. looks a bit ugly, but better this
  // way.

  if ( thepar ) {
    // propagate to it. assume that jacobian will be set correctly
    Gaudi::TrackMatrix  jac;
    Gaudi::TrackMatrix* pjac = m ? &jac : 0;
    double              z    = z1;
    sc                       = m_refextrapolator->propagate( in, z, thepar->plane1().z, m, geometry );
    z                        = thepar->plane1().z;

    if ( sc.isSuccess() ) {
      thepar->propagate( in, pjac );

      if ( m ) {
        Gaudi::TrackMatrix mtmp = ( *pjac ) * ( *m );
        *m                      = mtmp;
      }
      z  = thepar->plane2().z;
      sc = m_refextrapolator->propagate( in, z, z2, pjac, geometry );
      if ( m ) {
        Gaudi::TrackMatrix mtmp = ( *pjac ) * ( *m );
        *m                      = mtmp;
      }
    }

  } else {
    sc = m_refextrapolator->propagate( in, z1, z2, m, geometry );
  }

  return sc;
}
