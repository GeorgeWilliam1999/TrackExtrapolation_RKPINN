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
#include "GaudiKernel/PhysicalConstants.h"

#include <array>
#include <mutex>
#include <optional>
#include <sstream>
#include <vector>

#include "TrackFieldExtrapolatorBase.h"

// LHCb maths
#include "LHCbMath/EigenTypes.h"
#include "LHCbMath/FastRoots.h"

/**************************************************************************************

Implementation of ATLAS's STEP algorithm. (Lund, Bugge, Gavrilenko, and
Strandlie, https://inspirehep.net/record/819221).

This is the Runge-Kutta-Nystrom (RKN). The RKN algorithm is especially
suited for 2nd order differential equations. (The standard RK is for
first order.) It has important advantages compared to 5th order RK:

A. There are only 4 stages rather than 6.

B. The 2nd stage uses the same position as the 3rd step, such that
   they can use the same value for the field.

C. The last stage is close to the final solution, which means one can
  reuse the field lookup in the next step.

When combined this reduces the number of field lookups from 6 to 2 per
step. Since the number of stages is smaller, there is also less
math. Finally, our old RK was implemented such that one could use many
different RK stage schemes: as a result it was full with
multiplications with 0. Those are now gone as well.

To use the RKN with adaptive step size, one needs an error
estimate. With 5th order RK this is the difference between the 5th and
4th order. However, RKN is only 2nd order. The ATLAS authors have
shown how to get an estimate of the error by using the 4 computed
stages. Having the error, we just use the same step adaptation
scheme that we used before (from Numerical Recipees).

*************************************************************************/

namespace {

  /// The default floating point precision to use with the Eigen types
  using FloatType = double;

  /// Basically just a wrapper around the Eigen class, but Zero default constructed...
  template <typename TYPE, int ROWS, int COLUMNS>
  class RKMatrix : public ::Eigen::Matrix<TYPE, ROWS, COLUMNS> {
    typedef ::Eigen::Matrix<TYPE, ROWS, COLUMNS> Base;

  public:
    /// Default constructor adds zero initialisation
    RKMatrix() : Base( Base::Zero() ) {}
    /// forward to base constructor
    using Base::Base;
  };

  /// Type for a 2-vector
  template <typename TYPE = FloatType>
  using RKVec2 = RKMatrix<TYPE, 2, 1>;

  /// Type for a 2 by 3 Matrix
  template <typename TYPE = FloatType>
  using RKMatrix23 = RKMatrix<TYPE, 2, 3>;

  /// Represenation of a State
  template <typename TYPE = FloatType>
  struct RKState final {
    RKState() = default;
    RKState( const FloatType _x, const FloatType _y,   //
             const FloatType _tx, const FloatType _ty, //
             const FloatType _qop, const FloatType _z )
        : xparameters( _x, _y ), txparameters( _tx, _ty ), qop( _qop ), z( _z ) {}
    RKVec2<TYPE> xparameters;
    RKVec2<TYPE> txparameters;

    FloatType       qop{ 0 };
    FloatType       z{ 0 };
    TYPE&           x() noexcept { return xparameters( 0 ); }
    TYPE&           y() noexcept { return xparameters( 1 ); }
    TYPE&           tx() noexcept { return txparameters( 0 ); }
    TYPE&           ty() noexcept { return txparameters( 1 ); }
    TYPE            tx() const noexcept { return txparameters( 0 ); }
    TYPE            ty() const noexcept { return txparameters( 1 ); }
    TYPE            x() const noexcept { return xparameters( 0 ); }
    TYPE            y() const noexcept { return xparameters( 1 ); }
    Gaudi::XYZPoint position() const noexcept { return { x(), y(), z }; }
  };

  template <typename TYPE = FloatType>
  struct RKStage final {
    RKState<TYPE>                           state;
    RKVec2<TYPE>                            derivative; // derivative (dtxdz,dtydz)
    TrackFieldExtrapolatorBase::FieldVector Bfield;
  };

  template <typename TYPE = FloatType>
  struct RKCache final {
    std::array<RKStage<TYPE>, 4> stage;
    int                          laststep{ -1 };
    int                          step{ 0 };
  };

  template <typename TYPE = FloatType>
  struct RKJacobian final {

    RKMatrix23<TYPE> xmatrix;
    RKMatrix23<TYPE> txmatrix;

    TYPE& dXdTx0() noexcept { return xmatrix( 0, 0 ); }
    TYPE& dYdTx0() noexcept { return xmatrix( 1, 0 ); }
    TYPE& dTxdTx0() noexcept { return txmatrix( 0, 0 ); }
    TYPE& dTydTx0() noexcept { return txmatrix( 1, 0 ); }

    TYPE& dXdTy0() noexcept { return xmatrix( 0, 1 ); }
    TYPE& dYdTy0() noexcept { return xmatrix( 1, 1 ); }
    TYPE& dTxdTy0() noexcept { return txmatrix( 0, 1 ); }
    TYPE& dTydTy0() noexcept { return txmatrix( 1, 1 ); }

    TYPE& dXdQoP0() noexcept { return xmatrix( 0, 2 ); }
    TYPE& dYdQoP0() noexcept { return xmatrix( 1, 2 ); }
    TYPE& dTxdQoP0() noexcept { return txmatrix( 0, 2 ); }
    TYPE& dTydQoP0() noexcept { return txmatrix( 1, 2 ); }

    TYPE dTxdTx0() const noexcept { return txmatrix( 0, 0 ); }
    TYPE dTydTx0() const noexcept { return txmatrix( 1, 0 ); }

    TYPE dTxdTy0() const noexcept { return txmatrix( 0, 1 ); }
    TYPE dTydTy0() const noexcept { return txmatrix( 1, 1 ); }

    TYPE dTxdQoP0() const noexcept { return txmatrix( 0, 2 ); }
    TYPE dTydQoP0() const noexcept { return txmatrix( 1, 2 ); }
  };

  struct RKStatistics final {
    RKStatistics() = default;
    RKStatistics& operator+=( const RKStatistics& rhs ) {
      minstep = std::min( minstep, rhs.minstep );
      maxstep = std::max( maxstep, rhs.maxstep );
      err += rhs.err;
      numstep += rhs.numstep;
      numfailedstep += rhs.numfailedstep;
      numincreasedstep += rhs.numincreasedstep;
      sumstep += rhs.sumstep;
      return *this;
    }
    double             sumstep{ 0 };
    double             minstep{ 1e9 };
    double             maxstep{ 0 };
    unsigned long long numstep{ 0 };
    unsigned long long numfailedstep{ 0 };
    unsigned long long numincreasedstep{ 0 };
    RKVec2<>           err;
  };

  RKVec2<> evaluateDerivatives( const RKState<>& state, const TrackFieldExtrapolatorBase::FieldVector& field ) {
    const auto tx  = state.tx();
    const auto ty  = state.ty();
    const auto qop = state.qop;

    const auto Bx = field.x();
    const auto By = field.y();
    const auto Bz = field.z();

    const auto tx2 = tx * tx;
    const auto ty2 = ty * ty;

    const auto qopnorm = qop * std::sqrt( 1.0 + tx2 + ty2 );
    const auto dtxdz   = qopnorm * ( ty * ( tx * Bx + Bz ) - ( 1.0 + tx2 ) * By );
    const auto dtydz   = qopnorm * ( -tx * ( ty * By + Bz ) + ( 1.0 + ty2 ) * Bx );

    return RKVec2<>( dtxdz, dtydz );
  }

  RKMatrix23<> evaluateDerivativesJacobian( const RKState<>& state, const RKJacobian<>& jacobian,
                                            const TrackFieldExtrapolatorBase::FieldVector& field ) {
    const auto tx  = state.tx();
    const auto ty  = state.ty();
    const auto qop = state.qop;

    const auto Bx = field.x();
    const auto By = field.y();
    const auto Bz = field.z();

    const auto tx2 = tx * tx;
    const auto ty2 = ty * ty;

    const auto n2 = 1.0 + tx2 + ty2;
    const auto n  = std::sqrt( n2 );

    const auto txBx = tx * Bx;
    const auto txBy = tx * By;
    const auto tyBy = ty * By;
    const auto tyBx = ty * Bx;

    const auto Ax = n * ( ty * ( txBx + Bz ) - ( 1 + tx2 ) * By );
    const auto Ay = n * ( -tx * ( tyBy + Bz ) + ( 1 + ty2 ) * Bx );

    const auto inv_n2 = 1.0 / n2;
    const auto Ax_n2  = Ax * inv_n2;
    const auto Ay_n2  = Ay * inv_n2;

    // now we compute 'dJacobian/dZ'
    const auto dAxdTx = Ax_n2 * tx + n * ( tyBx - 2 * txBy );
    const auto dAxdTy = Ax_n2 * ty + n * ( txBx + Bz );

    const auto dAydTx = Ay_n2 * tx + n * ( -tyBy - Bz );
    const auto dAydTy = Ay_n2 * ty + n * ( -txBy + 2 * tyBx );

    // we'll do the factors of c later

    RKMatrix23<> deriv;

    // derivatives to Tx0
    // jacobianderiv.dXdTx0()  = jacobian.dTxdTx0() ;
    // jacobianderiv.dYdTx0()  = jacobian.dTydTx0() ;
    deriv( 0, 0 ) = qop * ( jacobian.dTxdTx0() * dAxdTx + jacobian.dTydTx0() * dAxdTy );
    deriv( 1, 0 ) = qop * ( jacobian.dTxdTx0() * dAydTx + jacobian.dTydTx0() * dAydTy );
    // derivatives to Ty0
    // jacobianderiv.dXdTy0()  = jacobian.dTxdTy0() ;
    // jacobianderiv.dYdTy0()  = jacobian.dTydTy0() ;
    deriv( 0, 1 ) = qop * ( jacobian.dTxdTy0() * dAxdTx + jacobian.dTydTy0() * dAxdTy );
    deriv( 1, 1 ) = qop * ( jacobian.dTxdTy0() * dAydTx + jacobian.dTydTy0() * dAydTy );
    // derivatives to qopc
    // jacobianderiv.dXdQoP0()  = jacobian.dTxdQoP0() ;
    // jacobianderiv.dYdQoP0()  = jacobian.dTydQoP0() ;
    deriv( 0, 2 ) = Ax + qop * ( jacobian.dTxdQoP0() * dAxdTx + jacobian.dTydQoP0() * dAxdTy );
    deriv( 1, 2 ) = Ay + qop * ( jacobian.dTxdQoP0() * dAydTx + jacobian.dTydQoP0() * dAydTy );
    // return
    return deriv;
  }

} // namespace

// *********************************************************************************************************

class TrackSTEPExtrapolator : public TrackFieldExtrapolatorBase {
public:
  /// enums
  enum RKErrorCode { RKSuccess, RKOutOfTolerance, RKCurling, RKExceededMaxNumSteps };

  /// Constructor
  TrackSTEPExtrapolator( const std::string& type, const std::string& name, const IInterface* parent );

  /// initialize
  StatusCode finalize() override;

  using TrackExtrapolator::propagate;

  /// Propagate a state vector from zOld to zNew
  /// Transport matrix is calulated when transMat pointer is not NULL
  StatusCode propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew, Gaudi::TrackMatrix* transMat,
                        IGeometryInfo const& geometry, LHCb::Tr::PID pid = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

  // public methods that are not in the interface. used for debugging with the extrapolator tester

private:
  RKErrorCode extrapolate( const LHCb::Magnet::MagneticFieldGrid* grid, RKState<>& state, double zout,
                           RKJacobian<>* jacobian, std::vector<double>* stepvector = nullptr ) const;
  RKErrorCode extrapolateNumericalJacobian( const LHCb::Magnet::MagneticFieldGrid* grid, RKState<>& state, double zout,
                                            RKJacobian<>& jacobian ) const;

  void evaluateRKStep( const LHCb::Magnet::MagneticFieldGrid* grid, const double dz, RKState<>& pin, RKVec2<>& err,
                       RKCache<>& cache ) const;
  void evaluateRKStepJacobian( const double dz, RKJacobian<>& jacobian, const RKCache<>& cache ) const;

private:
  // tool properties
  Gaudi::Property<double> m_toleranceX{ this, "Tolerance", 0.005 * Gaudi::Units::mm };
  Gaudi::Property<double> m_minRKStep{ this, "MinStep", 10 * Gaudi::Units::mm };
  Gaudi::Property<double> m_maxRKStep{ this, "MaxStep", 1 * Gaudi::Units::m };
  Gaudi::Property<double> m_initialRKStep{ this, "InitialStep", 1 * Gaudi::Units::m };
  Gaudi::Property<double> m_sigma{ this, "Sigma", 5.5 };
  Gaudi::Property<double> m_minStepScale{ this, "MinStepScale", 0.125 };
  Gaudi::Property<double> m_maxStepScale{ this, "MaxStepScale", 4.0 };
  Gaudi::Property<double> m_safetyFactor{ this, "StepScaleSafetyFactor", 1.0 };
  Gaudi::Property<size_t> m_maxNumRKSteps{ this, "MaxNumSteps", 1000 };
  Gaudi::Property<bool>   m_numericalJacobian{ this, "NumericalJacobian", false };
  Gaudi::Property<double> m_maxSlope{ this, "MaxSlope", 10.0 };
  Gaudi::Property<double> m_maxCurvature{ this, "MaxCurvature", 1 / Gaudi::Units::m };
  Gaudi::Property<bool>   m_useFieldLastStep{ this, "UseFieldLastStep", true };

  // data cache
  double m_toleranceX_inv{ 0 }; ///< 1 / m_toleranceX

  // keep statistics for monitoring
  mutable unsigned long long m_numcalls{ 0 };
  mutable RKStatistics       m_totalstats; ///< sum of stats for all calls
  mutable std::mutex         m_updateLock; ///< update lock
};

DECLARE_COMPONENT( TrackSTEPExtrapolator )

/// TrackSTEPExtrapolator constructor.
TrackSTEPExtrapolator::TrackSTEPExtrapolator( const std::string& type, const std::string& name,
                                              const IInterface* parent )
    : TrackFieldExtrapolatorBase( type, name, parent ) {
  // keep m_toleranceX_inv in sync with 1.0/m_toleranceX
  m_toleranceX.declareUpdateHandler( [this]( const auto& ) //
                                     { this->m_toleranceX_inv = 1.0 / this->m_toleranceX.value(); } );
  m_toleranceX.useUpdateHandler();
}

StatusCode TrackSTEPExtrapolator::finalize() {
  if ( msgLevel( MSG::DEBUG ) && m_numcalls > 0 ) {
    debug() << "Number of calls:     " << m_numcalls << endmsg;
    debug() << "Min step length:     " << m_totalstats.minstep << endmsg;
    debug() << "Max step length:     " << m_totalstats.maxstep << endmsg;
    debug() << "Av step length:      " << m_totalstats.sumstep / ( m_totalstats.numstep - m_totalstats.numfailedstep )
            << endmsg;
    debug() << "Av num step:         " << m_totalstats.numstep / double( m_numcalls ) << endmsg;
    debug() << "Fr. failed steps:    " << m_totalstats.numfailedstep / double( m_totalstats.numstep ) << endmsg;
    debug() << "Fr. increased steps: " << m_totalstats.numincreasedstep / double( m_totalstats.numstep ) << endmsg;
  }

  return TrackFieldExtrapolatorBase::finalize();
}

// Propagate a state vector from zOld to zNew
// Transport matrix is calulated when transMat pointer is not NULL
StatusCode TrackSTEPExtrapolator::propagate( Gaudi::TrackVector& state, double zin, double zout,
                                             Gaudi::TrackMatrix* transMat, IGeometryInfo const&,
                                             const LHCb::Tr::PID /*pid*/,
                                             const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  // Bail out if already at destination
  if ( std::abs( zin - zout ) < TrackParameters::propagationTolerance ) {
    if ( transMat ) *transMat = ROOT::Math::SMatrixIdentity();
    return StatusCode::SUCCESS;
  }

  std::optional<RKJacobian<>> jacobian;
  if ( transMat != nullptr ) { jacobian.emplace(); }

  // translate the state to one we use in the runge kutta. note the factor c.
  RKState<> rkstate( state( 0 ), state( 1 ), state( 2 ), state( 3 ), state( 4 ) * Gaudi::Units::c_light, zin );

  StatusCode  sc = StatusCode::SUCCESS;
  RKErrorCode success =
      ( m_numericalJacobian && jacobian ? extrapolateNumericalJacobian( grid, rkstate, zout, *jacobian )
                                        : extrapolate( grid, rkstate, zout, jacobian ? &jacobian.value() : nullptr ) );
  if ( success == RKSuccess ) {
    // translate the state back
    // info() << "In  " << state(0) << " " << state(1) << " " << state(2) << " " << state(3) << endmsg;
    state( 0 ) = rkstate.x();
    state( 1 ) = rkstate.y();
    state( 2 ) = rkstate.tx();
    state( 3 ) = rkstate.ty();
    // info() << "Out " << state(0) << " " << state(1) << " " << state(2) << " " << state(3) << endmsg;

    if ( transMat != nullptr ) {
      *transMat             = Gaudi::TrackMatrix();
      ( *transMat )( 0, 0 ) = 1;
      ( *transMat )( 1, 1 ) = 1;
      ( *transMat )( 4, 4 ) = 1;
      GAUDI_LOOP_UNROLL( 2 )
      for ( int irow = 0; irow < 2; ++irow ) {
        GAUDI_LOOP_UNROLL( 3 )
        for ( int icol = 0; icol < 3; ++icol ) {
          ( *transMat )( irow, icol + 2 )     = jacobian->xmatrix( irow, icol );
          ( *transMat )( irow + 2, icol + 2 ) = jacobian->txmatrix( irow, icol );
        }
      }

      // put back the factor c
      GAUDI_LOOP_UNROLL( 4 )
      for ( int irow = 0; irow < 4; ++irow ) { ( *transMat )( irow, 4 ) *= Gaudi::Units::c_light; }
    }
  } else {
    sc = Warning( "STEPExtrapolator failed with code: " + std::to_string( success ), StatusCode::FAILURE, 0 );
  }
  return sc;
}

TrackSTEPExtrapolator::RKErrorCode //
TrackSTEPExtrapolator::extrapolate( const LHCb::Magnet::MagneticFieldGrid* grid, RKState<>& state, double zout,
                                    RKJacobian<>* jacobian, std::vector<double>* stepvector ) const {

  // initialize the jacobian
  if ( jacobian != nullptr ) {
    jacobian->dTxdTx0() = 1;
    jacobian->dTydTy0() = 1;
  }

  // now start stepping. first try with a single step. this may not be
  // very optimal inside the magnet.
  const auto totalStep = zout - state.z;
  // auto toleranceTx = std::abs(m_toleranceX/totalStep) ;
  // const auto toleranceX = m_toleranceX.value();
  // auto toleranceTx = toleranceX/std::abs(totalStep) ;

  auto       absstep   = std::min( std::abs( totalStep ), m_initialRKStep.value() );
  const auto direction = totalStep > 0 ? +1 : -1;
  bool       laststep  = absstep < m_minRKStep;

  RKCache<> rkcache;
  RKVec2<>  err;

  std::size_t numfailedstep{ 0 };

  const bool                  do_debug = msgLevel( MSG::DEBUG );
  std::optional<RKStatistics> stats;
  if ( do_debug ) { stats.emplace(); }

  // FIXME: If we also define a minimum step, then we can get rid of
  // the extrapolator selector and TrackParabolic extrapolator.
  RKErrorCode rc = RKSuccess;
  while ( rc == RKSuccess && std::abs( state.z - zout ) > TrackParameters::propagationTolerance ) {
    // verbose() << state.z << " " << absstep << " " << laststep << endmsg ;

    // make a single range-kutta step
    auto prevstate = state;
    evaluateRKStep( grid, absstep * direction, state, err, rkcache );

    // decide if the error is small enough

    // always accept the step if it is smaller than the minimum step size
    bool success = ( absstep <= m_minRKStep );
    if ( !success ) {
      // if ( m_correctNumSteps ) {
      //   const auto estimatedN = std::abs(totalStep) / absstep ;
      //   toleranceX  = (m_toleranceX/estimatedN/m_sigma) ;
      //   toleranceTx = toleranceX/std::abs(totalStep) ;
      //   //(m_toleranceX/10000)/estimatedN/m_sigma ;
      //}

      // apply the acceptance criterion.
      auto normdx = std::abs( err( 0 ) ) * m_toleranceX_inv;
      auto normdy = std::abs( err( 1 ) ) * m_toleranceX_inv;
      // auto deltatx = state.tx() - prevstate.tx() ;
      // auto normdtx = std::abs( err(2) ) / ( toleranceTx + std::abs( deltatx ) * m_relToleranceTx ) ;
      auto errorOverTolerance = std::max( normdx, normdy );
      success                 = ( errorOverTolerance <= m_sigma );
      //     std::cout << "step: " << rkcache.step << " " << success << " "
      //                 << prevstate.z << " "
      //                 << state.z << " " << absstep << " "
      //                 << errorOverTolerance << std::endl ;

      // do some stepping monitoring, before adapting step size
      if ( success ) {
        if ( do_debug ) {
          stats->sumstep += absstep;
          if ( !laststep ) stats->minstep = std::min( stats->minstep, absstep );
          stats->maxstep = std::max( stats->maxstep, absstep );
        }
      } else {
        ++numfailedstep;
      }

      // adapt the stepsize if necessary. the powers come from num.recipees.
      double stepfactor( m_maxStepScale );
      if ( errorOverTolerance > 1 ) { // decrease step size
        stepfactor = std::max( m_minStepScale.value(), m_safetyFactor / std::sqrt( std::sqrt( errorOverTolerance ) ) );
      } else { // increase step size
        if ( errorOverTolerance > 0 ) {
          stepfactor =
              std::min( m_maxStepScale.value(), m_safetyFactor * FastRoots::invfifthroot( errorOverTolerance ) );
        }
        if ( do_debug ) { ++( stats->numincreasedstep ); }
      }
      absstep *= stepfactor;

      // apply another limitation criterion
      absstep = std::max( m_minRKStep.value(), std::min( absstep, m_maxRKStep.value() ) );
    }

    // info() << "Success = " << success << endmsg;
    if ( success ) {
      // if we need the jacobian, evaluate it only for successful steps
      auto thisstep = state.z - prevstate.z; // absstep has already been changed!
      if ( jacobian ) evaluateRKStepJacobian( thisstep, *jacobian, rkcache );
      // update the step, to invalidate the cache (or reuse the last stage)
      ++rkcache.step;
      if ( stepvector ) stepvector->push_back( thisstep );
      if ( do_debug ) { stats->err += err; }
    } else {
      // if this step failed, don't update the state
      state = prevstate;
    }

    // check that we don't step beyond the target
    const auto z_diff = zout - state.z;
    if ( absstep - ( direction * z_diff ) > 0 ) {
      absstep  = std::abs( z_diff );
      laststep = true;
    }

    // final check: bail out for vertical or looping tracks
    if ( std::max( std::abs( state.tx() ), std::abs( state.ty() ) ) > m_maxSlope ) {
      if ( do_debug )
        debug() << "State has very large slope, probably curling: tx, ty = " << state.tx() << ", " << state.ty()
                << " z_origin, target, current: " << zout - totalStep << " " << zout << " " << state.z << endmsg;
      rc = RKCurling;
    } else if ( std::abs( state.qop * rkcache.stage[0].Bfield.y() ) > m_maxCurvature ) {
      if ( do_debug )
        debug() << "State has too small curvature radius: " << state.qop * rkcache.stage[0].Bfield.y()
                << " z_origin, target, current: " << zout - totalStep << " " << zout << " " << state.z << endmsg;
      rc = RKCurling;
    } else if ( ( numfailedstep + rkcache.step ) >= m_maxNumRKSteps ) {
      if ( do_debug ) debug() << "Exceeded max numsteps. " << endmsg;
      rc = RKExceededMaxNumSteps;
    }
  }

  if ( do_debug ) {
    // update mutable cached stats
    std::lock_guard lock( m_updateLock );
    ++m_numcalls;
    stats->numstep       = rkcache.step;
    stats->numfailedstep = numfailedstep;
    m_totalstats += *stats;
  }

  return rc;
}

void TrackSTEPExtrapolator::evaluateRKStep( const LHCb::Magnet::MagneticFieldGrid* grid, const double dz,
                                            RKState<>& pin, RKVec2<>& err, RKCache<>& cache ) const {
  // debug() << "z-component of input: "
  //<< pin.z << " " << dz << endmsg ;

  // compute the first step: k0 = f(z,x,t)
  // if previous step failed, reuse the first stage.
  auto& stage0 = cache.stage[0];
  if ( cache.laststep == cache.step ) {
    // firststage = 1 ;
    // k[0] = state0.derivative ;
    // assert( std::abs(pin.z - cache.stage[0].state.z) < 1e-4 ) ;
  } else {
    // shall we reuse the B field from the last stage of the previous step?
    stage0.state = pin;
    if ( cache.laststep > 0 && m_useFieldLastStep ) {
      stage0.Bfield = cache.stage[3].Bfield;
    } else {
      stage0.Bfield = fieldVector( grid, stage0.state.position() );
    }
    stage0.derivative = evaluateDerivatives( stage0.state, stage0.Bfield );
    cache.laststep    = cache.step;
  }

  const auto dzdz   = dz * dz;
  const auto halfdz = 0.5 * dz;

  // compute the second step
  auto& stage1 = cache.stage[1];
  stage1.state = pin;
  stage1.state.z += halfdz;
  stage1.state.xparameters += halfdz * pin.txparameters + 0.125 * dzdz * stage0.derivative;
  stage1.state.txparameters += halfdz * stage0.derivative;
  stage1.Bfield     = fieldVector( grid, stage1.state.position() );
  stage1.derivative = evaluateDerivatives( stage1.state, stage1.Bfield );

  // compute the third step: identical to second except for derivative of tx
  auto& stage2              = cache.stage[2];
  stage2                    = stage1;
  stage2.state.txparameters = pin.txparameters;
  stage2.state.txparameters += halfdz * stage1.derivative;
  stage2.derivative = evaluateDerivatives( stage2.state, stage2.Bfield );

  // compute the last step
  auto& stage3 = cache.stage[3];
  stage3.state = pin;
  stage3.state.z += dz;
  stage3.state.xparameters += dz * ( pin.txparameters + halfdz * stage2.derivative );
  stage3.state.txparameters += dz * stage2.derivative;
  stage3.Bfield     = fieldVector( grid, stage3.state.position() );
  stage3.derivative = evaluateDerivatives( stage3.state, stage3.Bfield );

  // update the state
  // FIXME: this is what is written in STEP paper, but it may be that order is exactly wrong. check with Numerical
  // Recipees!
  pin.xparameters += dz * pin.txparameters + dzdz / 6.0 * ( stage0.derivative + stage1.derivative + stage2.derivative );
  pin.txparameters +=
      dz / 6.0 * ( stage0.derivative + 2.0 * ( stage1.derivative + stage2.derivative ) + stage3.derivative );
  pin.z += dz;

  // now compute the error
  err = dzdz * ( stage0.derivative - stage1.derivative - stage2.derivative + stage3.derivative );
}

void TrackSTEPExtrapolator::evaluateRKStepJacobian( const double dz, RKJacobian<>& jacobian,
                                                    const RKCache<>& cache ) const {
  // evaluate the jacobian. note that we never reuse last stage
  // here. that's not entirely consistent (but who cares)
  // std::array< RKMatrix43<>, 4 > k; // # stages is at most 7 ( DormondPrince )

  // * first evaluate the derivatives of k[0...3] to (tx, ty, qop)_in
  // * then just apply the same update equations as above. but does that give the correct derivative for x?
  // * and finally 'multiply' with the existing jacobian, rather than just 'add up'
  // * to evaluate the (2x3) derivative of k_i to (tx0,ty0,qop)

  // *FIXME: there may be a bug in the math here. The jocabian comes out slightly
  // smaller than the one from the normal RK.

  // stage 0
  auto         jtmp = jacobian;
  RKMatrix23<> k0   = evaluateDerivativesJacobian( cache.stage[0].state, jtmp, cache.stage[0].Bfield );
  // stage 1
  jtmp.txmatrix   = jacobian.txmatrix + 0.5 * dz * k0;
  RKMatrix23<> k1 = evaluateDerivativesJacobian( cache.stage[1].state, jtmp, cache.stage[1].Bfield );
  // stage 2
  jtmp.txmatrix   = jacobian.txmatrix + 0.5 * dz * k1;
  RKMatrix23<> k2 = evaluateDerivativesJacobian( cache.stage[2].state, jtmp, cache.stage[2].Bfield );
  // stage 3
  jtmp.txmatrix   = jacobian.txmatrix + dz * k2;
  RKMatrix23<> k3 = evaluateDerivativesJacobian( cache.stage[2].state, jtmp, cache.stage[2].Bfield );

  // complete bullshit?
  jacobian.xmatrix += dz * jacobian.txmatrix + dz * dz / 6.0 * ( k0 + k1 + k2 );
  jacobian.txmatrix += dz / 6.0 * ( k0 + 2 * ( k1 + k2 ) + k3 );
}

TrackSTEPExtrapolator::RKErrorCode //
TrackSTEPExtrapolator::extrapolateNumericalJacobian( const LHCb::Magnet::MagneticFieldGrid* grid, RKState<>& state,
                                                     double zout, RKJacobian<>& jacobian ) const {

  RKState<>           inputstate( state );
  std::vector<double> stepvector;
  stepvector.reserve( 256 );
  RKErrorCode success = extrapolate( grid, state, zout, &jacobian, &stepvector );
  if ( success == RKSuccess ) {
    // now make small changes in tx,ty,qop
    double delta[3] = { 0.01, 0.01, 1e-8 };
    for ( int col = 0; col < 3; ++col ) {
      RKState<> astate( inputstate );
      switch ( col ) {
      case 0:
        astate.tx() += delta[0];
        break;
      case 1:
        astate.ty() += delta[1];
        break;
      case 2:
        astate.qop += delta[2];
        break;
      }
      RKCache<> cache;
      RKVec2<>  err;
      for ( size_t j = 0; j < stepvector.size(); ++j ) {
        evaluateRKStep( grid, stepvector[j], astate, err, cache );
        ++cache.step;
      }
      if ( !( std::abs( state.z - astate.z ) < TrackParameters::propagationTolerance ) ) {
        std::ostringstream mess;
        mess << "problem in numerical integration."
             << " zin: " << inputstate.z << " zout: " << zout << " state.z: " << state.z << " dstate.z: " << astate.z;
        Warning( mess.str() ).ignore();
      }
      assert( std::abs( state.z - astate.z ) < TrackParameters::propagationTolerance );

      GAUDI_LOOP_UNROLL( 2 )
      for ( int row = 0; row < 2; ++row ) {
        const auto inv                = 1.0 / delta[col];
        jacobian.xmatrix( row, col )  = ( astate.xparameters( row ) - state.xparameters( row ) ) * inv;
        jacobian.txmatrix( row, col ) = ( astate.txparameters( row ) - state.txparameters( row ) ) * inv;
      }
    }
  }
  return success;
}
