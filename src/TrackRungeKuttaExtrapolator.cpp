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
#include "Kernel/SynchronizedValue.h"
#include "LHCbMath/EigenTypes.h"
#include "LHCbMath/FastRoots.h"
#include "TrackFieldExtrapolatorBase.h"
#include <array>
#include <atomic>
#include <execution>
#include <iomanip>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "Event/States.h"

namespace RK {
  /// enums
  enum class ErrorCode : StatusCode::code_t {
    Success        = 1,
    OutOfTolerance = 10,
    Curling,
    ExceededMaxNumSteps,
    BadDistance
  };
  struct ErrorCategory : StatusCode::Category {
    const char* name() const override { return "RungeKuttaExtrapolator"; }
    bool        isRecoverable( StatusCode::code_t ) const override { return false; }
    std::string message( StatusCode::code_t code ) const override {
      switch ( static_cast<ErrorCode>( code ) ) {
      case ErrorCode::OutOfTolerance:
        return "RK: Out of tollerance ";
      case ErrorCode::Curling:
        return "RK: Curling";
      case ErrorCode::ExceededMaxNumSteps:
        return "RK: Exceed maximum number of steps";
      case ErrorCode::BadDistance:
        return "RK: Very large value (> 1e6 mm) for state position. Breaking iteration";
      default:
        return StatusCode::default_category().message( code );
      }
    }
  };
} // namespace RK
STATUSCODE_ENUM_DECL( RK::ErrorCode )
STATUSCODE_ENUM_IMPL( RK::ErrorCode, RK::ErrorCategory )

namespace RK {
  class ErrorCounters {
  public:
    template <typename Component>
    ErrorCounters( Component* parent, const char* prefix = "", const std::size_t nMax = 0ul )
        : m_out{ parent, prefix + StatusCode{ ErrorCode::OutOfTolerance }.ignore().message(), nMax }
        , m_curl{ parent, prefix + StatusCode{ ErrorCode::Curling }.ignore().message(), nMax }
        , m_max{ parent, prefix + StatusCode{ ErrorCode::ExceededMaxNumSteps }.ignore().message(), nMax }
        , m_dist{ parent, prefix + StatusCode{ ErrorCode::BadDistance }.ignore().message(), nMax } {}

    ErrorCounters( const ErrorCounters& )            = delete;
    ErrorCounters( ErrorCounters&& )                 = delete;
    ErrorCounters& operator=( const ErrorCounters& ) = delete;
    ErrorCounters& operator=( ErrorCounters&& )      = delete;

    ErrorCounters& operator+=( RK::ErrorCode code ) {
      switch ( code ) {
      case ErrorCode::OutOfTolerance:
        ++m_out;
        break;
      case ErrorCode::Curling:
        ++m_curl;
        break;
      case ErrorCode::ExceededMaxNumSteps:
        ++m_max;
        break;
      case ErrorCode::BadDistance:
        ++m_dist;
        break;
      case ErrorCode::Success:
        break;
      }
      return *this;
    }

  private:
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_out;
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_curl;
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_max;
    mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_dist;
  };
  namespace Scheme {

    // note: CashKarp appears twice, to flag it is as being the 'default' value...
    meta_enum_class_with_unknown( scheme_t, unsigned char, CashKarp, CashKarp, Fehlberg, DormandPrice, BogackiShampine3,
                                  BogackiShampine5, HeunEuler, SharpSmart5, Tsitouras5, Tsitouras2009, SharpVerner6,
                                  SharpVerner7, Verner6, Verner7, Verner8, Verner9 )

  } // namespace Scheme

  using Scheme::scheme_t;

  namespace details {

    /// The default floating point precision to use with the Eigen types
    using FloatType = double;

    /// Basically just a wrapper around the Eigen class, but Zero default constructed...
    template <typename TYPE, int ROWS, int COLUMNS>
    class Matrix : public ::Eigen::Matrix<TYPE, ROWS, COLUMNS> {
      typedef ::Eigen::Matrix<TYPE, ROWS, COLUMNS> Base;

    public:
      /// Default constructor adds zero initialisation
      Matrix() : Base( Base::Zero() ) {}
      /// forward to base constructor
      using Base::Base;
    };

    /// Type for a 4-vector
    template <typename TYPE = FloatType>
    using Vec4 = Matrix<TYPE, 4, 1>;

    /// Type for a 4 by 3 Matrix
    template <typename TYPE = FloatType>
    using Matrix43 = Matrix<TYPE, 4, 3>;

    /// Represenation of a State
    template <typename TYPE = FloatType>
    struct State final {
      State() = default;
      State( Vec4<TYPE> _vec, const FloatType _qop, const FloatType _z )
          : parameters( std::move( _vec ) ), qop( _qop ), z( _z ) {}
      Vec4<TYPE> parameters;
      FloatType  qop{ 0 };
      FloatType  z{ 0 };
      TYPE&      x() noexcept { return parameters( 0 ); }
      TYPE&      y() noexcept { return parameters( 1 ); }
      TYPE&      tx() noexcept { return parameters( 2 ); }
      TYPE&      ty() noexcept { return parameters( 3 ); }
      TYPE       tx() const noexcept { return parameters( 2 ); }
      TYPE       ty() const noexcept { return parameters( 3 ); }
    };

    template <typename TYPE = FloatType>
    struct Stage final {
      State<TYPE>                             state;
      State<TYPE>                             derivative;
      TrackFieldExtrapolatorBase::FieldVector Bfield;
    };

    template <unsigned nStages, typename TYPE = FloatType>
    struct Cache final {
      std::array<Stage<TYPE>, nStages> stage;
      int                              laststep{ -1 };
      int                              step{ 0 };
    };

    template <typename TYPE = FloatType>
    struct Jacobian final {

      Matrix43<TYPE> matrix;

      TYPE& dXdTx0() noexcept { return matrix( 0, 0 ); }
      TYPE& dYdTx0() noexcept { return matrix( 1, 0 ); }
      TYPE& dTxdTx0() noexcept { return matrix( 2, 0 ); }
      TYPE& dTydTx0() noexcept { return matrix( 3, 0 ); }

      TYPE& dXdTy0() noexcept { return matrix( 0, 1 ); }
      TYPE& dYdTy0() noexcept { return matrix( 1, 1 ); }
      TYPE& dTxdTy0() noexcept { return matrix( 2, 1 ); }
      TYPE& dTydTy0() noexcept { return matrix( 3, 1 ); }

      TYPE& dXdQoP0() noexcept { return matrix( 0, 2 ); }
      TYPE& dYdQoP0() noexcept { return matrix( 1, 2 ); }
      TYPE& dTxdQoP0() noexcept { return matrix( 2, 2 ); }
      TYPE& dTydQoP0() noexcept { return matrix( 3, 2 ); }

      TYPE dTxdTx0() const noexcept { return matrix( 2, 0 ); }
      TYPE dTydTx0() const noexcept { return matrix( 3, 0 ); }

      TYPE dTxdTy0() const noexcept { return matrix( 2, 1 ); }
      TYPE dTydTy0() const noexcept { return matrix( 3, 1 ); }

      TYPE dTxdQoP0() const noexcept { return matrix( 2, 2 ); }
      TYPE dTydQoP0() const noexcept { return matrix( 3, 2 ); }
    };

  } // namespace details

  struct Statistics final {
    Statistics& operator+=( const Statistics& rhs ) {
      minstep = std::min( minstep, rhs.minstep );
      maxstep = std::max( maxstep, rhs.maxstep );
      err += rhs.err;
      numstep += rhs.numstep;
      numfailedstep += rhs.numfailedstep;
      numincreasedstep += rhs.numincreasedstep;
      sumstep += rhs.sumstep;
      return *this;
    }
    double          sumstep{ 0 };
    double          minstep{ 1e9 };
    double          maxstep{ 0 };
    size_t          numstep{ 0 };
    size_t          numfailedstep{ 0 };
    size_t          numincreasedstep{ 0 };
    details::Vec4<> err;
  };

  // *********************************************************************************************************
  // Butcher tables for various adaptive Runge Kutta methods. These are all taken from wikipedia.
  // *********************************************************************************************************
  namespace Scheme {

    template <unsigned N, bool firstSameAsLast>
    struct ButcherTableau {
      static constexpr unsigned                       NStages = N;
      std::array<double, NStages*( NStages - 1 ) / 2> a;
      std::array<double, NStages>                     b;
      std::array<double, NStages>                     b_star; // lower order approx
    };

    namespace {

      struct Rat {
        std::intmax_t num = 0, den = 1;
        constexpr operator double() const noexcept { return static_cast<double>( num ) / static_cast<double>( den ); }
        constexpr explicit operator bool() const noexcept { return num != 0; }
      };

      // https://github.com/SciML/DiffEqDevTools.jl/blob/master/src/ode_tableaus.jl
      // https://docs.sciml.ai/v1.6/solvers/ode_solve.html

      constexpr auto HeunEuler =
          ButcherTableau<2, false>{ { Rat{ 1 } }, { Rat{ 1 }, Rat{ 0 } }, { Rat{ 1, 2 }, Rat{ 1, 2 } } };

      constexpr auto CashKarp = ButcherTableau<6, false>{
          { Rat{ 1, 5 },                                                //
            Rat{ 3, 40 }, Rat{ 9, 40 },                                 //
            Rat{ 3, 10 }, Rat{ -9, 10 }, Rat{ 6, 5 },                   //
            Rat{ -11, 54 }, Rat{ 5, 2 }, Rat{ -70, 27 }, Rat{ 35, 27 }, //
            Rat{ 1631, 55296 }, Rat{ 175, 512 }, Rat{ 575, 13824 }, Rat{ 44275, 110592 }, Rat{ 253, 4096 } },
          { Rat{ 37, 378 }, Rat{ 0 }, Rat{ 250, 621 }, Rat{ 125, 594 }, Rat{ 0 }, Rat{ 512, 1771 } },
          { Rat{ 2825, 27648 }, Rat{ 0 }, Rat{ 18575, 48384 }, Rat{ 13525, 55296 }, Rat{ 277, 14336 }, Rat{ 1, 4 } } };

      constexpr auto Fehlberg = ButcherTableau<6, false>{
          { Rat{ 1, 4 },                                                     //
            Rat{ 3, 32 }, Rat{ 9, 32 },                                      //
            Rat{ 1932, 2197 }, Rat{ -7200, 2197 }, Rat{ 7296, 2197 },        //
            Rat{ 439, 216 }, Rat{ -8 }, Rat{ 3680, 513 }, Rat{ -845, 4104 }, //
            Rat{ -8, 27 }, Rat{ 2 }, Rat{ -3544, 2565 }, Rat{ 1859, 4104 }, Rat{ -11, 40 } },
          { Rat{ 16, 135 }, Rat{ 0 }, Rat{ 6656, 12825 }, Rat{ 28561, 56430 }, Rat{ -9, 50 }, Rat{ 2, 55 } },
          { Rat{ 25, 216 }, Rat{ 0 }, Rat{ 1408, 2565 }, Rat{ 2197, 4104 }, Rat{ -1, 5 }, Rat{ 0 } },
      };

      // https://doi.org/10.1016/0771-050X(80)90013-3
      constexpr auto DormandPrice =
          ButcherTableau<7, true>{ { Rat{ 1, 5 },                       //
                                     Rat{ 3, 40 },        Rat{ 9, 40 }, //
                                     Rat{ 44, 45 },       Rat{ -56, 15 },
                                     Rat{ 32, 9 }, //
                                     Rat{ 19372, 6561 },  Rat{ -25360, 2187 },
                                     Rat{ 64448, 6561 },  Rat{ -212, 729 }, //
                                     Rat{ 9017, 3168 },   Rat{ -355, 33 },
                                     Rat{ 46732, 5247 },  Rat{ 49, 176 },
                                     Rat{ -5103, 18656 }, //
                                     Rat{ 35, 384 },      Rat{ 0 },
                                     Rat{ 500, 1113 },    Rat{ 125, 192 },
                                     Rat{ -2187, 6784 },  Rat{ 11, 84 } },
                                   { Rat{ 5179, 57600 }, Rat{ 0 }, Rat{ 7571, 16695 }, Rat{ 393, 640 },
                                     Rat{ -92097, 339200 }, Rat{ 187, 2100 }, Rat{ 1, 40 } },
                                   { Rat{ 35, 384 }, Rat{ 0 }, Rat{ 500, 1113 }, Rat{ 125, 192 }, Rat{ -2187, 6784 },
                                     Rat{ 11, 84 }, Rat{ 0 } } };

      // https://doi.org/10.1016%2F0893-9659%2889%2990079-7
      constexpr auto BogackiShampine3 =
          ButcherTableau<4, false>{ { Rat{ 1, 2 }, Rat{ 0 }, Rat{ 3, 4 }, Rat{ 2, 9 }, Rat{ 1, 3 }, Rat{ 4, 9 } },
                                    { Rat{ 7, 24 }, Rat{ 1, 4 }, Rat{ 1, 3 }, Rat{ 1, 8 } },
                                    { Rat{ 2, 9 }, Rat{ 1, 3 }, Rat{ 4, 9 }, Rat{ 0 } } };

      constexpr auto BogackiShampine5 =
          ButcherTableau<8, false>{ { Rat{ 1, 6 },
                                      Rat{ 2, 27 },
                                      Rat{ 4, 27 },
                                      Rat{ 183, 1372 },
                                      Rat{ -162, 343 },
                                      Rat{ 1053, 1372 },
                                      Rat{ 68, 297 },
                                      Rat{ -4, 11 },
                                      Rat{ 42, 143 },
                                      Rat{ 1960, 3861 },
                                      Rat{ 597, 22528 },
                                      Rat{ 81, 352 },
                                      Rat{ 63099, 585728 },
                                      Rat{ 58653, 366080 },
                                      Rat{ 4617, 20480 },
                                      Rat{ 174197, 959244 },
                                      Rat{ -30942, 79937 },
                                      Rat{ 8152137, 19744439 },
                                      Rat{ 666106, 1039181 },
                                      Rat{ -29421, 29068 },
                                      Rat{ 482048, 414219 },
                                      Rat{ 587, 8064 },
                                      Rat{ 0 },
                                      Rat{ 4440339, 15491840 },
                                      Rat{ 24353, 124800 },
                                      Rat{ 387, 44800 },
                                      Rat{ 2152, 5985 },
                                      Rat{ 7267, 94080 } },
                                    { Rat{ 587, 8064 }, Rat{ 0 }, Rat{ 4440339, 15491840 }, Rat{ 24353, 124800 },
                                      Rat{ 387, 44800 }, Rat{ 2152, 5985 }, Rat{ 7267, 94080 }, Rat{ 0 } },
                                    { Rat{ 6059, 80640 }, Rat{ 0 }, Rat{ 8559189, 30983680 }, Rat{ 26411, 124800 },
                                      Rat{ -927, 89600 }, Rat{ 443, 1197 }, Rat{ 7267, 94080 }, Rat{ 0 } } };

      constexpr auto SharpSmart5 = ButcherTableau<7, false>{
          { Rat{ 16, 105 },
            Rat{ 2, 35 },
            Rat{ 6, 35 },
            Rat{ 8793, 40960 },
            Rat{ -5103, 8192 },
            Rat{ 17577, 20480 },
            Rat{ 347, 1458 },
            Rat{ -7, 20 },
            Rat{ 3395, 10044 },
            Rat{ 49792, 112995 },
            Rat{ -1223224109959, 9199771214400 },
            Rat{ 1234787701, 2523942720 },
            Rat{ 568994101921, 3168810084960 },
            Rat{ -105209683888, 891227836395 },
            Rat{ 9, 25 },
            Rat{ 2462504862877, 8306031988800 },
            Rat{ -123991, 287040 },
            Rat{ 106522578491, 408709510560 },
            Rat{ 590616498832, 804646848915 },
            Rat{ -319138726, 534081275 },
            Rat{ 52758, 71449 } },
          { Rat{ 1093, 15120 }, Rat{ 0 }, Rat{ 60025, 190992 }, Rat{ 3200, 20709 }, Rat{ 1611, 11960 },
            Rat{ 712233, 2857960 }, Rat{ 3, 40 } },
          { Rat{ 84018211, 991368000 }, Rat{ 0 }, Rat{ 92098979, 357791680 }, Rat{ 17606944, 67891005 },
            Rat{ 3142101, 235253200 }, Rat{ 22004596809, 70270091500 }, Rat{ 9, 125 } } };

      //
      // Sharp-Verner Order 5/6 method
      // Completely Imbedded Runge-Kutta Pairs, by P. W. Sharp and J. H. Verner,
      //  SIAM Journal on Numerical Analysis, Vol. 31, No. 4. (Aug., 1994), pages. 1169 to 1190.

      constexpr auto SharpVerner6 =
          ButcherTableau<9, true>{ { Rat{ 1, 12 },
                                     Rat{ 2, 75 },
                                     Rat{ 8, 75 },
                                     Rat{ 1, 20 },
                                     Rat{ 0 },
                                     Rat{ 3, 20 },
                                     Rat{ 88, 135 },
                                     Rat{ 0 },
                                     Rat{ -112, 45 },
                                     Rat{ 64, 27 },
                                     Rat{ -10891, 11556 },
                                     Rat{ 0 },
                                     Rat{ 3880, 963 },
                                     Rat{ -8456, 2889 },
                                     Rat{ 217, 428 },
                                     Rat{ 1718911, 4382720 },
                                     Rat{ 0 },
                                     Rat{ -1000749, 547840 },
                                     Rat{ 819261, 383488 },
                                     Rat{ -671175, 876544 },
                                     Rat{ 14535, 14336 },
                                     Rat{ 85153, 203300 },
                                     Rat{ 0 },
                                     Rat{ -6783, 2140 },
                                     Rat{ 10956, 2675 },
                                     Rat{ -38493, 13375 },
                                     Rat{ 1152, 425 },
                                     Rat{ -7168, 40375 },
                                     Rat{ 53, 912 },
                                     Rat{ 0 },
                                     Rat{ 0 },
                                     Rat{ 5, 16 },
                                     Rat{ 27, 112 },
                                     Rat{ 27, 136 },
                                     Rat{ 256, 969 },
                                     Rat{ -25, 336 } },
                                   { Rat{ 53, 912 }, Rat{ 0 }, Rat{ 0 }, Rat{ 5, 16 }, Rat{ 27, 112 }, Rat{ 27, 136 },
                                     Rat{ 256, 969 }, Rat{ -25, 336 }, Rat{ 0 } },
                                   { Rat{ 617, 10944 }, Rat{ 0 }, Rat{ 0 }, Rat{ 241, 756 }, Rat{ 69, 320 },
                                     Rat{ 435, 1904 }, Rat{ 10304, 43605 }, Rat{ 0 }, Rat{ -1, 18 } } };

      // Completely Imbedded Runge-Kutta Pairs, by P.W.Sharp and J.H.Verner, Siam Journal on Numerical Analysis, Vol.31,
      // No.4. (August 1994) pages 1169-1190.  https://doi.org/10.1137/0731061
      constexpr auto SharpVerner7 = ButcherTableau<12, true>{
          { Rat{ 1, 12 },
            Rat{ 4, 243 },
            Rat{ 32, 243 },
            Rat{ 1, 18 },
            Rat{ 0 },
            Rat{ 1, 6 },
            Rat{ 5, 9 },
            Rat{ 0 },
            Rat{ -25, 12 },
            Rat{ 25, 12 },
            Rat{ 1, 15 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 1, 3 },
            Rat{ 4, 15 },
            Rat{ 319, 3840 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 161, 1536 },
            Rat{ -41, 960 },
            Rat{ 11, 512 },
            Rat{ 245, 5184 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 1627, 10368 },
            Rat{ 151, 1296 },
            Rat{ -445, 10368 },
            Rat{ 1, 6 },
            Rat{ -556349853, 7539261440 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ -4356175383, 3015704576 },
            Rat{ -814787343, 1884815360 },
            Rat{ 831004641, 3015704576 },
            Rat{ 355452237, 235601920 },
            Rat{ 107943759, 117800960 },
            Rat{ -68998698967, 1063035863040 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ -767387292485, 425214345216 },
            Rat{ -205995991597, 265758965760 },
            Rat{ -22181208863, 141738115072 },
            Rat{ 26226796959, 15502606336 },
            Rat{ 1614200643, 1107329024 },
            Rat{ 187, 329 },
            Rat{ 24511479161, 17979371520 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 3889847115, 217931776 },
            Rat{ 22028391, 3681280 },
            Rat{ 614528179, 217931776 },
            Rat{ -148401247, 10215552 },
            Rat{ -3122234829, 318384704 },
            Rat{ -4160, 1221 },
            Rat{ 15040, 20757 },
            Rat{ 5519, 110880 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 83, 560 },
            Rat{ 932, 3675 },
            Rat{ 282123, 1047200 },
            Rat{ 2624, 24255 },
            Rat{ 3008, 19635 },
            Rat{ 37, 2100 } },
          { Rat{ 5519, 110880 }, Rat{ 0 }, Rat{ 0 }, Rat{ 0 }, Rat{ 0 }, Rat{ 83, 560 }, Rat{ 932, 3675 },
            Rat{ 282123, 1047200 }, Rat{ 2624, 24255 }, Rat{ 3008, 19635 }, Rat{ 37, 2100 } },
          { Rat{ 15509, 341880 }, Rat{ 0 }, Rat{ 0 }, Rat{ 0 }, Rat{ 0 }, Rat{ 6827, 15540 }, Rat{ 22138, 81585 },
            Rat{ 78003, 387464 }, Rat{ -64144, 299145 }, Rat{ 623408, 2179485 }, Rat{ 0 }, Rat{ -15, 518 }

          } };

      constexpr auto Verner6 = ButcherTableau<8, false>{
          { Rat{ 1, 7 },           Rat{ 4, 81 },         Rat{ 14, 81 },        Rat{ 291, 1372 },
            Rat{ -27, 49 },        Rat{ 1053, 1372 },    Rat{ 86, 297 },       Rat{ -14, 33 },
            Rat{ 42, 143 },        Rat{ 1960, 3861 },    Rat{ -267, 22528 },   Rat{ 189, 704 },
            Rat{ 63099, 585728 },  Rat{ 58653, 366080 }, Rat{ 4617, 20480 },   Rat{ 10949, 6912 },
            Rat{ -69, 32 },        Rat{ -90891, 68096 }, Rat{ 112931, 25920 }, Rat{ -69861, 17920 },
            Rat{ 26378, 10773 },   Rat{ 1501, 19008 },   Rat{ -21, 88 },       Rat{ 219519, 347776 },
            Rat{ 163807, 926640 }, Rat{ -417, 640 },     Rat{ 1544, 1539 },    Rat{ 0 } },
          { Rat{ 79, 1080 }, Rat{ 0 }, Rat{ 19683, 69160 }, Rat{ 16807, 84240 }, Rat{ 0 }, Rat{ 2816, 7695 },
            Rat{ 1, 100 }, Rat{ 187, 2800 } },
          { Rat{ 763, 10800 }, Rat{ 0 }, Rat{ 59049, 197600 }, Rat{ 88837, 526500 }, Rat{ 243, 4000 },
            Rat{ 12352, 38475 }, Rat{ 0 }, Rat{ 2, 25 } } };

      // http://www.mymathlib.com/diffeq/
      // http://www.mymathlib.com/diffeq/embedded_runge_kutta/embedded_verner_7_8.html
      constexpr auto Verner7 = ButcherTableau<10, false>{
          { .5e-2,
            -1.076790123456790123456790123456790123457,
            1.185679012345679012345679012345679012346,
            .4083333333333333333333333333333333333333e-1,
            0.,
            .1225,
            .6360714285714285714285714285714285714286,
            0.,
            -2.444464285714285714285714285714285714286,
            2.263392857142857142857142857142857142857,
            -2.535121107934924522925638355466021548721,
            0.,
            10.29937465444926792043851446075602491361,
            -7.951303288599057994949321745826687653648,
            .7930114892310059220122601427111526182380,
            1.001876581252463296196919658309499980821,
            0.,
            -4.166571282442379833131393800547097145319,
            3.834343292912864241255266521825137866520,
            -.5023333356071084754746433022861176561240,
            .6676847438841607711538509226985769541026,
            27.25501835463076713033396381917500571735,
            0.,
            -42.00461727841063835531864544390929536961,
            -10.53571312661948991792108160054652610372,
            80.49553671141193714798365215892682663420,
            -67.34388227179051346854907596321297564093,
            13.04865761077793746347118702956696476271,
            -3.039737805711496514694365865875576322688,
            0.,
            10.13816141032980111185794619070970015044,
            -6.429305674864721572146282562955529806444,
            -1.586437148340827658711531285379861057947,
            1.892178184196842441086430890913135336502,
            .1969933540760886906129236016333644283801e-1,
            .5441698982793323546510272424795257297790e-2,
            -1.444951891677773513735100317935571236052,
            0.,
            8.031891385995591922411703322301956043504,
            -7.583174166340134682079888302367158860498,
            3.581616935319007421124768544245287869686,
            -2.436972263219952941118380906569375238373,
            .8515899999232617933968976603248614217339,
            0.,
            0. },
          { .4742583783370675608356917271757453469893e-1, 0., 0., .2562236165937056265996172745827462344816,
            .2695137683307420661947381725807595288676, .1268662240909278284598913836473917324788,
            .2488722594206007162204644942764749276729, .3074483740820063133530438847909918476864e-2,
            .4802380998949694330818906334714312332321e-1, 0. },
          {
              .4748524769929963103753127380572796155227e-1,
              0.,
              0.,
              .2559941258869063329715491824590539387050,
              .2705847808106768872253089109926813573239,
              .1250561868442599291363882232374691792045,
              .2520446872374386050718404382019744256218,
              0.,
              0.,
              .4883497152141861455738197130309313759259e-1,
          } };

      constexpr auto Verner8 = ButcherTableau<13, false>{
          { Rat{ 1, 4 },
            Rat{ 5, 72 },
            Rat{ 1, 72 },
            Rat{ 1, 32 },
            Rat{ 0 },
            Rat{ 3, 32 },
            Rat{ 106, 125 },
            Rat{ 0 },
            Rat{ -408, 125 },
            Rat{ 352, 125 },
            Rat{ 1, 48 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 8, 33 },
            Rat{ 125, 528 },
            Rat{ -1263, 2401 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 39936, 26411 },
            Rat{ -64125, 26411 },
            Rat{ 5520, 2401 },
            Rat{ 37, 392 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 1625, 9408 },
            Rat{ -2, 15 },
            Rat{ 61, 6720 },
            Rat{ 17176, 25515 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ -47104, 25515 },
            Rat{ 1325, 504 },
            Rat{ -41792, 25515 },
            Rat{ 20237, 145800 },
            Rat{ 4312, 6075 },
            Rat{ -23834, 180075 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ -77824, 1980825 },
            Rat{ -636635, 633864 },
            Rat{ 254048, 300125 },
            Rat{ -183, 7000 },
            Rat{ 8, 11 },
            Rat{ -324, 3773 },
            Rat{ 12733, 7600 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ -20032, 5225 },
            Rat{ 456485, 80256 },
            Rat{ -42599, 7125 },
            Rat{ 339227, 912000 },
            Rat{ -1029, 4180 },
            Rat{ 1701, 1408 },
            Rat{ 5145, 2432 },
            Rat{ -27061, 204120 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ 40448, 280665 },
            Rat{ -1353775, 1197504 },
            Rat{ 17662, 25515 },
            Rat{ -71687, 1166400 },
            Rat{ 98, 225 },
            Rat{ 1, 16 },
            Rat{ 3773, 11664 },
            Rat{ 0 },
            Rat{ 11203, 8680 },
            Rat{ 0 },
            Rat{ 0 },
            Rat{ -38144, 11935 },
            Rat{ 2354425, 458304 },
            Rat{ -84046, 16275 },
            Rat{ 673309, 1636800 },
            Rat{ 4704, 8525 },
            Rat{ 9477, 10912 },
            Rat{ -1029, 992 },
            Rat{ 0 },
            Rat{ 729, 341 } },
          { Rat{ 31, 720 }, 0, 0, 0, 0, Rat{ 16, 75 }, Rat{ 16807, 79200 }, Rat{ 16807, 79200 }, Rat{ 243, 1760 }, 0, 0,
            Rat{ 243, 1760 }, Rat{ 31, 720 } },
          { Rat{ 13, 288 }, 0, 0, 0, 0, Rat{ 32, 125 }, Rat{ 31213, 144000 }, Rat{ 2401, 12375 }, Rat{ 1701, 14080 },
            Rat{ 2401, 19200 }, Rat{ 19, 450 }, Rat{ 0 }, Rat{ 0 } }

      };

      // http://people.math.sfu.ca/~jverner/RKV98.IIa.Robust.000000351.081209.FLOAT6040OnWeb
      constexpr auto Verner9 = ButcherTableau<16, false>{
          { .40e-1,
            -.198852731918229097650241511466089129345e-1,
            .1163726333296965222173745449432724803716,
            .3618276005170260466963139767374883778890e-1,
            0.,
            .1085482801551078140088941930212465133667,
            2.272114264290177409193144938921415409241,
            0.,
            -8.526886447976398578316416192982602292786,
            6.830772183686221169123271254061186883545,
            .5094385535389374394512668566783434123978e-1,
            0.,
            0.,
            .1755865049809071110203693328749561646990,
            .70229612707574674987780067603244497535e-3,
            .1424783668683284782770955365543878809824,
            0.,
            0.,
            -.3541799434668684104094753917518523845155,
            .7595315450295100889001534202778550159932e-1,
            .6765157656337123215269906939508560510196,
            .7111111111111111111111111111111111111111e-1,
            0.,
            0.,
            0.,
            0.,
            .3279909287605898328568406057725491803016,
            .2408979601282990560320482831163397085872,
            .7125e-1,
            0.,
            0.,
            0.,
            0.,
            .3268842451575245554847578757216915662785,
            .1156157548424754445152421242783084337215,
            -.3375e-1,
            .4822677322465810178387112087673611111111e-1,
            0.,
            0.,
            0.,
            0.,
            .3948559980495400110769549704186108167677e-1,
            .1058851161934658144373823566907778072121,
            -.2152006320474309346664428710937500000000e-1,
            -.1045374260183348238623046875000000000000,
            -.2609113435754923412210928689962011065179e-1,
            0.,
            0.,
            0.,
            0.,
            .3333333333333333333333333333333333333333e-1,
            -.1652504006638105086724681598195267241410,
            .3434664118368616658319419895678838776647e-1,
            .1595758283215209043195814910843067811951,
            .2140857321828193385584684233447183324979,
            -.362842339625565859076509979091267105528e-1,
            0.,
            0.,
            0.,
            0.,
            -1.096167597427208807028761474420297770752,
            .1826035504321331052308236240517254331348,
            .708225444417068325613028685455625123741e-1,
            -.231364701848243126999929738482630407146e-1,
            .2711204726320932916455631550463654973432,
            1.308133749422980744437146904349994472286,
            -.5074635056416974879347823927726392374259,
            0.,
            0.,
            0.,
            0.,
            -6.631342198657237090355284142048733580937,
            -.252748010090880105270020973014860316405,
            -.4952612380036095562991116175550167835424,
            .293252554525388690285739720360003594753,
            1.440108693768280908474851998204423941413,
            6.237934498647055877243623886838802127716,
            .7270192054526987638549835199880202544289,
            .6130118256955931701496387847232542148725,
            0.,
            0.,
            0.,
            0.,
            9.088803891640463313341034206647776279557,
            -.407378815629344868103315381138325162923,
            1.790733389490374687043894756399015035977,
            .714927166761755073724875250629602731782,
            -1.438580857841722850237810322456327208949,
            -8.263329312064740580595954649844133476994,
            -1.537570570808865115231450725068826856201,
            .3453832827564871699090880801079644428793,
            -1.211697910343873872490625222495537087293,
            0.,
            0.,
            0.,
            0.,
            -19.05581871559595277753334676575234493500,
            1.26306067538987510135943101851905310045,
            -6.913916969178458046793476128409110926069,
            -.676462266509498065300115641383621209887,
            3.367860445026607887090352785684064242560,
            18.00675164312590810020103216906571965203,
            6.838828926794279896350389904990814350968,
            -1.031516451921950498420447675652291096155,
            .4129106232130622755368055554332539084021,
            2.157389007494053627033175177985666660692,
            0.,
            0.,
            0.,
            0.,
            23.80712219809580523172312179815279712750,
            .88627792492165554903036801415266308369,
            13.13913039759876381480201677314222971522,
            -2.604415709287714883747369630937415176632,
            -5.193859949783872300189266203049579105962,
            -20.41234071154150778768154893536134356354,
            -12.30085625250572261314889445241581039623,
            1.521553095008539362178397458330791655267,
            0.,
            0. },
          { .1458885278405539719101539582255752917034e-1, 0., 0., 0., 0., 0., 0.,
            .2024197887889332650566666683195656097825e-2, .2178047084569716646796256135839225745895,
            .1274895340854389692868677968654808668201, .2244617745463131861258531547137348031621,
            .1787254491259903095100090833796054447157, .7594344758096557172908303416513173076283e-1,
            .1294845879197561516869001434704642286297, .2947744761261941714007911131590716605202e-1, 0. },
          { .2034666655224434599707885098832906986649e-1, 0., 0., 0., 0., 0., 0.,
            1.069617650982700109541321983413338230042, .7680834711303187278673130261850350530338e-1,
            .1130778186885240437498706751119241126785, .2552587357981962194892445789565762186511,
            -.9825898086919164036191607912120918904022, .3981545824421514217762002137442675068982, 0., 0.,
            .4932600711506839027871318637915324696208e-1 } };

      // Runge-Kutta pairs of orders 5(4) using the minimal set of simplifying assumptions,
      // by Ch. Tsitouras, TEI of Chalkis, Dept. of Applied Sciences, GR34400, Psahna, Greece.
      // constants taken from https://github.com/SciML/DiffEqDevTools.jl/blob/master/src/ode_tableaus.jl
      constexpr auto Tsitouras5 = ButcherTableau<7, true>{
          { Rat{ 161, 1000 },
            -.8480655492356988544426874250230774675121177393430391537369234245294192976164141156943e-2,
            .3354806554923569885444268742502307746751211773934303915373692342452941929761641411569,
            2.897153057105493432130432594192938764924887287701866490314866693455023795137503079289,
            -6.359448489975074843148159912383825625952700647415626703305928850207288721235210244366,
            4.362295432869581411017727318190886861027813359713760212991062156752264926097707165077,
            5.325864828439256604428877920840511317836476253097040101202360397727981648835607691791,
            -11.74888356406282787774717033978577296188744178259862899288666928009020615663593781589,
            7.495539342889836208304604784564358155658679161518186721010132816213648793440552049753,
            -.9249506636175524925650207933207191611349983406029535244034750452930469056411389539635e-1,
            5.861455442946420028659251486982647890394337666164814434818157239052507339770711679748,
            -12.92096931784710929170611868178335939541780751955743459166312250439928519268343184452,
            8.159367898576158643180400794539253485181918321135053305748355423955009222648673734986,
            -.7158497328140099722453054252582973869127213147363544882721139659546372402303777878835e-1,
            -.2826905039406838290900305721271224146717633626879770007617876201276764571291579142206e-1,
            .9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1,
            Rat{ 1, 100 },
            .4798896504144995747752495322905965199130404621990332488332634944254542060153074523509,
            1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331,
            -3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677,
            2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841 },
          { .9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1, Rat{ 1, 100 },
            .4798896504144995747752495322905965199130404621990332488332634944254542060153074523509,
            1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331,
            -3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677,
            2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841 },
          { .9468075576583945807478876255758922856117527357724631226139574065785592789071067303271e-1,
            .9183565540343253096776363936645313759813746240984095238905939532922955247253608687270e-2,
            .4877705284247615707855642599631228241516691959761363774365216240304071651579571959813,
            1.234297566930478985655109673884237654035539930748192848315425833500484878378061439761,
            -2.707712349983525454881109975059321670689605166938197378763992255714444407154902012702,
            1.866628418170587035753719399566211498666255505244122593996591602841258328965767580089, Rat{ 1, 66 } } };

      // Runge-Kutta Pairs of Orders 5(4) using the Minimal Set of Simplifying Assumptions
      // AIP Conference Proceedings 1168, 69 (2009), https://doi.org/10.1063/1.3241561
      constexpr auto Tsitouras2009 = ButcherTableau<7, false>{
          { 0.231572163526079, 0.271356352139396, -0.059103796886580, 0.043071565237093, 4.560080615554683,
            -4.006458683473722, 0.084777789595161, -2.443935658802774, 2.631461258707441, 0.524706566208284,
            0.072257770735164, 9.516251378071800, -8.467630087008555, -0.987888827522473, 0.867009765724064 },
          { 0.091937670648056, 1.156529958312496, -0.781330409541651, 0.197624776163019, 0.271639883438847,
            0.063598120979232, 0 },
          { 0.092167469090589, 1.131750860603267, -0.759749304413104, 0.205573577541223, 0.264767065074229,
            0.040490332103796, Rat{ 1, 40 } } };
    } // namespace

    template <typename Callable>
    auto invoke_with_tableau( Callable f, scheme_t s ) {
      switch ( s ) {
      case scheme_t::CashKarp:
        return f( CashKarp );
      case scheme_t::Fehlberg:
        return f( Fehlberg );
      case scheme_t::DormandPrice:
        return f( DormandPrice );
      case scheme_t::BogackiShampine3:
        return f( BogackiShampine3 );
      case scheme_t::BogackiShampine5:
        return f( BogackiShampine5 );
      case scheme_t::HeunEuler:
        return f( HeunEuler );
      case scheme_t::SharpSmart5:
        return f( SharpSmart5 );
      case scheme_t::Tsitouras5:
        return f( Tsitouras5 );
      case scheme_t::Tsitouras2009:
        return f( Tsitouras2009 );
      case scheme_t::SharpVerner6:
        return f( SharpVerner6 );
      case scheme_t::SharpVerner7:
        return f( SharpVerner7 );
      case scheme_t::Verner6:
        return f( Verner6 );
      case scheme_t::Verner7:
        return f( Verner7 );
      case scheme_t::Verner8:
        return f( Verner8 );
      case scheme_t::Verner9:
        return f( Verner9 );
      }
      throw std::domain_error{ "invalid RK::scheme_t" };
    }

    namespace {
      details::State<> evaluateDerivatives( const details::State<>&                        state,
                                            const TrackFieldExtrapolatorBase::FieldVector& field ) {
        const auto tx  = state.tx();
        const auto ty  = state.ty();
        const auto qop = state.qop;

        const auto tx2 = tx * tx;
        const auto ty2 = ty * ty;

        const auto Bx = field.x();
        const auto By = field.y();
        const auto Bz = field.z();

        const auto norm = std::sqrt( 1 + tx2 + ty2 );
        const auto Ax   = norm * ( ty * ( tx * Bx + Bz ) - ( 1 + tx2 ) * By );
        const auto Ay   = norm * ( -tx * ( ty * By + Bz ) + ( 1 + ty2 ) * Bx );

        // this is 'dState/Dz'
        //       x   y   tx  ty   qop  z
        return { { tx, ty, qop * Ax, qop * Ay }, 0, 1 };
      }

      details::Jacobian<> evaluateDerivativesJacobian( const details::State<>&                        state,
                                                       const details::Jacobian<>&                     jacobian,
                                                       const TrackFieldExtrapolatorBase::FieldVector& field ) {
        const auto tx  = state.tx();
        const auto ty  = state.ty();
        const auto qop = state.qop;

        const auto Bx = field.x();
        const auto By = field.y();
        const auto Bz = field.z();

        const auto tx2 = tx * tx;
        const auto ty2 = ty * ty;

        const auto n2 = 1 + tx2 + ty2;
        const auto n  = std::sqrt( n2 );

        const auto txBx = tx * Bx;
        const auto txBy = tx * By;
        const auto tyBy = ty * By;
        const auto tyBx = ty * Bx;

        const auto Ax = n * ( ty * ( txBx + Bz ) - ( 1 + tx2 ) * By );
        const auto Ay = n * ( -tx * ( tyBy + Bz ) + ( 1 + ty2 ) * Bx );

        const auto Ax_n2 = Ax / n2;
        const auto Ay_n2 = Ay / n2;

        // now we compute 'dJacobian/dZ'
        const auto dAxdTx = Ax_n2 * tx + n * ( tyBx - 2 * txBy );
        const auto dAxdTy = Ax_n2 * ty + n * ( txBx + Bz );

        const auto dAydTx = Ay_n2 * tx + n * ( -tyBy - Bz );
        const auto dAydTy = Ay_n2 * ty + n * ( -txBy + 2 * tyBx );

        // we'll do the factors of c later
        details::Jacobian<> jacobianderiv;

        // derivatives to Tx0
        jacobianderiv.dXdTx0()  = jacobian.dTxdTx0();
        jacobianderiv.dYdTx0()  = jacobian.dTydTx0();
        jacobianderiv.dTxdTx0() = qop * ( jacobian.dTxdTx0() * dAxdTx + jacobian.dTydTx0() * dAxdTy );
        jacobianderiv.dTydTx0() = qop * ( jacobian.dTxdTx0() * dAydTx + jacobian.dTydTx0() * dAydTy );

        // derivatives to Ty0
        jacobianderiv.dXdTy0()  = jacobian.dTxdTy0();
        jacobianderiv.dYdTy0()  = jacobian.dTydTy0();
        jacobianderiv.dTxdTy0() = qop * ( jacobian.dTxdTy0() * dAxdTx + jacobian.dTydTy0() * dAxdTy );
        jacobianderiv.dTydTy0() = qop * ( jacobian.dTxdTy0() * dAydTx + jacobian.dTydTy0() * dAydTy );

        // derivatives to qopc
        jacobianderiv.dXdQoP0()  = jacobian.dTxdQoP0();
        jacobianderiv.dYdQoP0()  = jacobian.dTydQoP0();
        jacobianderiv.dTxdQoP0() = Ax + qop * ( jacobian.dTxdQoP0() * dAxdTx + jacobian.dTydQoP0() * dAxdTy );
        jacobianderiv.dTydQoP0() = Ay + qop * ( jacobian.dTxdQoP0() * dAydTx + jacobian.dTydQoP0() * dAydTy );
        return jacobianderiv;
      }
    } // namespace

    template <unsigned NStages, bool firstSameAsLast, typename FieldVector>
    RK::ErrorCode evaluateStep( const ButcherTableau<NStages, firstSameAsLast>& table, double dz, details::State<>& pin,
                                details::Vec4<>& err, details::Cache<NStages>& cache, FieldVector fieldVector ) {

      std::array<details::Vec4<>, NStages> k;
      int                                  firststage( 0 );

      // previous step failed, reuse the first stage
      if ( cache.laststep == cache.step ) {
        firststage = 1;
        k[0]       = dz * cache.stage[0].derivative.parameters;
        // assert( std::abs(pin.z - cache.stage[0].state.z) < 1e-4 ) ;
      }
      // previous step succeeded and we can reuse the last stage (Dormand-Price)
      else if ( firstSameAsLast && cache.step > 0 ) {
        firststage     = 1;
        cache.stage[0] = cache.stage[NStages - 1];
        k[0]           = dz * cache.stage[0].derivative.parameters;
      }
      cache.laststep = cache.step;

      for ( int m = firststage; m != NStages; ++m ) {
        auto& stage = cache.stage[m];
        // evaluate the state
        stage.state = std::transform_reduce(
            std::execution::unseq, begin( k ), std::next( begin( k ), m ),
            std::next( begin( table.a ), m * ( m - 1 ) / 2 ), pin,
            []( details::State<> lhs, const details::State<>& rhs ) {
              // TODO: what about qop?
              lhs.parameters += rhs.parameters;
              lhs.z += rhs.z;
              return lhs;
            },
            [dz]( const details::Vec4<>& rhs, double a ) -> details::State<> {
              return { a * rhs, 0, a * dz };
            } );
        constexpr auto bad_distance = 100 * Gaudi::Units::meter;
        if ( std::abs( stage.state.x() ) > bad_distance || std::abs( stage.state.y() ) > bad_distance ||
             std::abs( stage.state.z ) > bad_distance ) {
          return RK::ErrorCode::BadDistance;
        }
        // evaluate the derivatives
        stage.Bfield     = fieldVector( { stage.state.x(), stage.state.y(), stage.state.z } );
        stage.derivative = evaluateDerivatives( stage.state, stage.Bfield );
        k[m]             = dz * stage.derivative.parameters;
      }

      // update state and error
      err.fill( 0 );
      for ( int m = 0; m != NStages; ++m ) {
        // this is the difference between the 4th and 5th order
        err += ( table.b[m] - table.b_star[m] ) * k[m];
        // this is the fifth order change
        pin.parameters += table.b[m] * k[m];
      }
      pin.z += dz;
      return RK::ErrorCode::Success;
    }

    template <unsigned NStages, bool fsl>
    void evaluateStepJacobian( const ButcherTableau<NStages, fsl>& table, double dz, details::Jacobian<>& jacobian,
                               const details::Cache<NStages>& cache ) {

      // evaluate the jacobian. not that we never resue last stage
      // here. that's not entirely consistent (but who cares)
      std::array<details::Matrix43<>, NStages> k;
      if constexpr ( NStages == 6 ) {
        // evaluate the derivatives. reuse the parameters and bfield from the cache
        k[0].noalias() =
            dz * evaluateDerivativesJacobian( cache.stage[0].state, jacobian, cache.stage[0].Bfield ).matrix;
        k[1].noalias() = dz * evaluateDerivativesJacobian( cache.stage[1].state,
                                                           details::Jacobian<>{ jacobian.matrix + table.a[0] * k[0] },
                                                           cache.stage[1].Bfield )
                                  .matrix;
        k[2].noalias() = dz * evaluateDerivativesJacobian(
                                  cache.stage[2].state,
                                  details::Jacobian<>{ jacobian.matrix + table.a[1] * k[0] + table.a[2] * k[1] },
                                  cache.stage[2].Bfield )
                                  .matrix;
        k[3].noalias() = dz * evaluateDerivativesJacobian( cache.stage[3].state,
                                                           details::Jacobian<>{ jacobian.matrix + table.a[3] * k[0] +
                                                                                table.a[4] * k[1] + table.a[5] * k[2] },
                                                           cache.stage[3].Bfield )
                                  .matrix;
        k[4].noalias() = dz * evaluateDerivativesJacobian( cache.stage[4].state,
                                                           details::Jacobian<>{ jacobian.matrix + table.a[6] * k[0] +
                                                                                table.a[7] * k[1] + table.a[8] * k[2] +
                                                                                table.a[9] * k[3] },
                                                           cache.stage[4].Bfield )
                                  .matrix;
        k[5].noalias() = dz * evaluateDerivativesJacobian(
                                  cache.stage[5].state,
                                  details::Jacobian<>{ jacobian.matrix + table.a[10] * k[0] + table.a[11] * k[1] +
                                                       table.a[12] * k[2] + table.a[13] * k[3] + table.a[14] * k[4] },
                                  cache.stage[5].Bfield )
                                  .matrix;
      } else {
        for ( int m = 0; m != NStages; ++m ) {
          // evaluate the derivatives. reuse the parameters and bfield from the cache
          k[m].noalias() =
              dz * evaluateDerivativesJacobian( cache.stage[m].state,
                                                std::inner_product(
                                                    begin( k ), std::next( begin( k ), m ),
                                                    std::next( begin( table.a ), m * ( m - 1 ) / 2 ), jacobian,
                                                    []( details::Jacobian<> j, details::Matrix43<> const& wk ) {
                                                      j.matrix += wk;
                                                      return j;
                                                    },
                                                    []( const details::Matrix43<>& ki, double a ) { return a * ki; } ),
                                                cache.stage[m].Bfield )
                       .matrix;
        }
      }

      for ( int m = 0; m != NStages; ++m ) { jacobian.matrix += table.b[m] * k[m]; }
    }
  } // namespace Scheme
} // namespace RK

class TrackRungeKuttaExtrapolator : public TrackFieldExtrapolatorBase {

public:
  /// Constructor
  using TrackFieldExtrapolatorBase::TrackFieldExtrapolatorBase;

  /// initialize
  StatusCode initialize() override;

  /// initialize
  StatusCode finalize() override;

  using TrackFieldExtrapolatorBase::propagate;

  /// Propagate a state vector from zOld to zNew
  /// Transport matrix is calulated when transMat pointer is not NULL
  StatusCode propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew, Gaudi::TrackMatrix* transMat,
                        IGeometryInfo const& geometry, const LHCb::Tr::PID pid = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

private:
  StatusCode propagate( RK::details::State<>& rkstate, double zNew, Gaudi::TrackMatrix* transMat,
                        IGeometryInfo const& geometry, const LHCb::Tr::PID pid = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const;

  struct doNothing {
    constexpr void operator()( double ) const {}
  };

  template <typename Tableau, typename StepObserver = doNothing>
  RK::ErrorCode extrapolate( const LHCb::Magnet::MagneticFieldGrid* grid, const Tableau& butcher,
                             RK::details::State<>& state, double zout, RK::details::Jacobian<>* jacobian,
                             StepObserver = {} ) const;
  template <typename Tableau>
  RK::ErrorCode extrapolateNumericalJacobian( const LHCb::Magnet::MagneticFieldGrid* grid, const Tableau& butcher,
                                              RK::details::State<>& state, double zout,
                                              RK::details::Jacobian<>& jacobian ) const;

private:
  // tool properties
  Gaudi::Property<double> m_toleranceX{ this, "Tolerance",
                                        0.001 * Gaudi::Units::mm };         ///< required absolute position resolution
  Gaudi::Property<double> m_relToleranceTx{ this, "RelToleranceTx", 5e-5 }; ///< required relative curvature resolution
  Gaudi::Property<double> m_minRKStep{ this, "MinStep", 10 * Gaudi::Units::mm };
  Gaudi::Property<double> m_maxRKStep{ this, "MaxStep", 1 * Gaudi::Units::m };
  Gaudi::Property<double> m_initialRKStep{ this, "InitialStep", 1 * Gaudi::Units::m };
  Gaudi::Property<double> m_sigma{ this, "Sigma", 5.5 };
  Gaudi::Property<double> m_minStepScale{ this, "MinStepScale", 0.125 };
  Gaudi::Property<double> m_maxStepScale{ this, "MaxStepScale", 4.0 };
  Gaudi::Property<double> m_safetyFactor{ this, "StepScaleSafetyFactor", 1.0 };
  Gaudi::Property<RK::scheme_t> m_rkscheme{ this, "RKScheme", RK::scheme_t::CashKarp };
  Gaudi::Property<size_t>       m_maxNumRKSteps{ this, "MaxNumSteps", 1000 };
  Gaudi::Property<bool>         m_correctNumSteps{ this, "CorrectNumSteps", false };
  Gaudi::Property<bool>         m_numericalJacobian{ this, "NumericalJacobian", false };
  Gaudi::Property<double>       m_maxSlope{ this, "MaxSlope", 10. };
  Gaudi::Property<double>       m_maxCurvature{ this, "MaxCurvature", 1 / Gaudi::Units::m };

  // keep statistics for monitoring
  mutable std::atomic<unsigned long long>               m_numcalls{ 0 };
  mutable LHCb::cxx::SynchronizedValue<RK::Statistics>  m_totalstats; // sum of stats for all calls
  mutable RK::ErrorCounters                             m_errors{ this, "RungeKuttaExtrapolator failed with code: " };
  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_numerical_integration_problem{
      this, "problem in numerical integration" };
};

//
//       (   1   0   dXdTx0   dXdTy0     dXdQoP0   )
//       (   0   1   dYdTx0   dYdTy0     dYdQoP0   )
//   J = (   0   0   dTxdTx0  dTxdTy0    dTxdQoP0  )
//       (   0   0   dTydTx0  dTxdTy0    dTydQoP0  )
//       (   0   0      0        0         1       )
//

namespace {
  template <typename T>
  const T* get_ptr( const std::optional<T>& opt ) {
    return opt ? &( *opt ) : nullptr;
  }
  template <typename T>
  T* get_ptr( std::optional<T>& opt ) {
    return opt ? &( *opt ) : nullptr;
  }

} // namespace

// *********************************************************************************************************

DECLARE_COMPONENT( TrackRungeKuttaExtrapolator )

StatusCode TrackRungeKuttaExtrapolator::finalize() {
  if ( msgLevel( MSG::DEBUG ) ) {
    debug() << "Number of calls:     " << m_numcalls << endmsg;
    m_totalstats.with_lock( [&]( const RK::Statistics& s ) {
      debug() << "Min step length:     " << s.minstep << endmsg;
      debug() << "Max step length:     " << s.maxstep << endmsg;
      debug() << "Av step length:      " << s.sumstep / ( s.numstep - s.numfailedstep ) << endmsg;
      debug() << "Av num step:         " << s.numstep / double( m_numcalls ) << endmsg;
      debug() << "Fr. failed steps:    " << s.numfailedstep / double( s.numstep ) << endmsg;
      debug() << "Fr. increased steps: " << s.numincreasedstep / double( s.numstep ) << endmsg;
    } );
  }
  return TrackFieldExtrapolatorBase::finalize();
}

StatusCode TrackRungeKuttaExtrapolator::initialize() {
  return TrackFieldExtrapolatorBase::initialize().andThen( [&] {
    if ( msgLevel( MSG::DEBUG ) ) debug() << "Using RK scheme: " << m_rkscheme.value() << endmsg;
    m_totalstats.with_lock( []( RK::Statistics& s ) { s = {}; } );
    m_numcalls = 0;
  } );
}

StatusCode TrackRungeKuttaExtrapolator::propagate( Gaudi::TrackVector& state, double zin, double zout,
                                                   Gaudi::TrackMatrix* transMat, IGeometryInfo const& geometry,
                                                   const LHCb::Tr::PID                    pid,
                                                   const LHCb::Magnet::MagneticFieldGrid* grid ) const {

  // Bail out if already at destination
  if ( std::abs( zin - zout ) < TrackParameters::propagationTolerance ) {
    if ( transMat ) *transMat = ROOT::Math::SMatrixIdentity();
    return StatusCode::SUCCESS;
  }

  RK::details::State<> rkstate( { state( 0 ), state( 1 ), state( 2 ), state( 3 ) }, state( 4 ) * Gaudi::Units::c_light,
                                zin );
  auto                 status = propagate( rkstate, zout, transMat, geometry, pid, grid );

  // translate the state back
  state( 0 ) = rkstate.x();
  state( 1 ) = rkstate.y();
  state( 2 ) = rkstate.tx();
  state( 3 ) = rkstate.ty();

  return status;
}

// Propagate a state vector from zOld to zNew
// Transport matrix is calulated when transMat pointer is not NULL
StatusCode TrackRungeKuttaExtrapolator::propagate( RK::details::State<>& rkstate, double zout,
                                                   Gaudi::TrackMatrix* transMat, IGeometryInfo const&,
                                                   const LHCb::Tr::PID /*pid*/,
                                                   const LHCb::Magnet::MagneticFieldGrid* grid ) const {

  std::optional<RK::details::Jacobian<>> jacobian;
  if ( transMat ) jacobian.emplace();

  RK::ErrorCode status = invoke_with_tableau(
      [&]( const auto& tableau ) {
        return ( m_numericalJacobian && jacobian
                     ? extrapolateNumericalJacobian( grid, tableau, rkstate, zout, *jacobian )
                     : extrapolate( grid, tableau, rkstate, zout, get_ptr( jacobian ) ) );
      },
      m_rkscheme );

  if ( status != RK::ErrorCode::Success ) {
    m_errors += status;
    return status;
  }

  if ( transMat ) {
    *transMat             = Gaudi::TrackMatrix();
    ( *transMat )( 0, 0 ) = 1;
    ( *transMat )( 1, 1 ) = 1;
    for ( int irow = 0; irow < 4; ++irow ) {
      for ( int icol = 0; icol < 3; ++icol ) ( *transMat )( irow, icol + 2 ) = jacobian->matrix( irow, icol );
      ( *transMat )( irow, 4 ) *= Gaudi::Units::c_light;
    }
    ( *transMat )( 4, 4 ) = 1;
  }

  return status;
}

template <typename Tableau, typename StepObserver>
RK::ErrorCode TrackRungeKuttaExtrapolator::extrapolate( const LHCb::Magnet::MagneticFieldGrid* grid,
                                                        const Tableau& table, RK::details::State<>& state, double zout,
                                                        RK::details::Jacobian<>* jacobian,
                                                        StepObserver             stepObserver ) const {
  // count calls
  ++m_numcalls;

  // initialize the jacobian
  if ( jacobian ) {
    jacobian->dTxdTx0() = 1;
    jacobian->dTydTy0() = 1;
  }

  // now start stepping. first try with a single step. this may not be
  // very optimal inside the magnet.
  const auto totalStep = zout - state.z;
  // auto toleranceTx = std::abs(m_toleranceX/totalStep) ;
  auto toleranceX  = m_toleranceX.value();
  auto toleranceTx = toleranceX / std::abs( totalStep );

  auto       absstep   = std::min( std::abs( totalStep ), m_initialRKStep.value() );
  const auto direction = totalStep > 0 ? +1 : -1;
  bool       laststep  = absstep < m_minRKStep;

  RK::details::Cache<Tableau::NStages> rkcache;
  RK::details::Vec4<>                  err, totalErr;
  RK::Statistics                       stats;
  RK::ErrorCode                        rc = RK::ErrorCode::Success;

  while ( rc == RK::ErrorCode::Success && std::abs( state.z - zout ) > TrackParameters::propagationTolerance ) {

    // make a single range-kutta step
    auto prevstate = state;
    if ( rc = evaluateStep( table, absstep * direction, state, err, rkcache,
                            [&]( const Gaudi::XYZPoint& position ) { return this->fieldVector( grid, position ); } );
         rc != RK::ErrorCode::Success ) {
      break;
    }

    // decide if the error is small enough

    // always accept the step if it is smaller than the minimum step size
    bool success = ( absstep <= m_minRKStep );
    if ( !success ) {
      if ( m_correctNumSteps ) {
        const auto estimatedN = std::abs( totalStep ) / absstep;
        toleranceX            = ( m_toleranceX / estimatedN / m_sigma );
        toleranceTx           = toleranceX / std::abs( totalStep );
        //(m_toleranceX/10000)/estimatedN/m_sigma ;
      }

      // apply the acceptance criterion.
      auto normdx             = std::abs( err( 0 ) ) / toleranceX;
      auto normdy             = std::abs( err( 1 ) ) / toleranceX;
      auto deltatx            = state.tx() - prevstate.tx();
      auto normdtx            = std::abs( err( 2 ) ) / ( toleranceTx + std::abs( deltatx ) * m_relToleranceTx );
      auto errorOverTolerance = std::max( normdx, std::max( normdy, normdtx ) );
      success                 = ( errorOverTolerance <= m_sigma );
      //     std::cout << "step: " << rkcache.step << " " << success << " "
      //                 << prevstate.z << " "
      //                 << state.z << " " << absstep << " "
      //                 << errorOverTolerance << std::endl ;

      // do some stepping monitoring, before adapting step size
      if ( success ) {
        stats.sumstep += absstep;
        if ( !laststep ) stats.minstep = std::min( stats.minstep, absstep );
        stats.maxstep = std::max( stats.maxstep, absstep );
      } else {
        ++stats.numfailedstep;
      }

      // adapt the stepsize if necessary. the powers come from num.recipees.
      double stepfactor( 1 );
      if ( errorOverTolerance > 1 ) { // decrease step size
        stepfactor = std::max( m_minStepScale.value(),
                               m_safetyFactor / std::sqrt( std::sqrt( errorOverTolerance ) ) ); // was : * std::pow(
                                                                                                // errorOverTolerance ,
                                                                                                // -0.25 )  ;
      } else {                                                                                  // increase step size
        if ( errorOverTolerance > 0 ) {
          stepfactor =
              std::min( m_maxStepScale.value(),
                        m_safetyFactor * FastRoots::invfifthroot( errorOverTolerance ) ); // was: * std::pow(
                                                                                          // errorOverTolerance,
                                                                                          // -0.2)  ;
        } else {
          stepfactor = m_maxStepScale;
        }
        ++stats.numincreasedstep;
      }

      // apply another limitation criterion
      absstep = std::max( m_minRKStep.value(), std::min( absstep * stepfactor, m_maxRKStep.value() ) );
    }

    // info() << "Success = " << success << endmsg;
    if ( success ) {
      // if we need the jacobian, evaluate it only for successful steps
      auto thisstep = state.z - prevstate.z; // absstep has already been changed!
      if ( jacobian ) evaluateStepJacobian( table, thisstep, *jacobian, rkcache );
      // update the step, to invalidate the cache (or reuse the last stage)
      ++rkcache.step;
      stepObserver( thisstep );
      stats.err += err;
    } else {
      // if this step failed, don't update the state
      state = prevstate;
    }

    // check that we don't step beyond the target
    if ( absstep > direction * ( zout - state.z ) ) {
      absstep  = std::abs( zout - state.z );
      laststep = true;
    }

    // final check: bail out for vertical or looping tracks
    if ( std::max( std::abs( state.tx() ), std::abs( state.ty() ) ) > m_maxSlope ) {
      if ( msgLevel( MSG::DEBUG ) )
        debug() << "State has very large slope, probably curling: tx, ty = " << state.tx() << ", " << state.ty()
                << " z_origin, target, current: " << zout - totalStep << " " << zout << " " << state.z << endmsg;
      rc = RK::ErrorCode::Curling;
    } else if ( std::abs( state.qop * rkcache.stage[0].Bfield.y() ) > m_maxCurvature ) {
      if ( msgLevel( MSG::DEBUG ) )
        debug() << "State has too small curvature radius: " << state.qop * rkcache.stage[0].Bfield.y()
                << " z_origin, target, current: " << zout - totalStep << " " << zout << " " << state.z << endmsg;
      rc = RK::ErrorCode::Curling;
    } else if ( stats.numfailedstep + rkcache.step >= m_maxNumRKSteps ) {
      if ( msgLevel( MSG::DEBUG ) ) debug() << "Exceeded max numsteps. " << endmsg;
      rc = RK::ErrorCode::ExceededMaxNumSteps;
    }
  }
  stats.numstep = rkcache.step;
  if ( msgLevel( MSG::DEBUG ) ) {
    m_totalstats.with_lock( [&]( RK::Statistics& s ) { s += stats; } );
  }
  return rc;
}

template <typename Tableau>
RK::ErrorCode TrackRungeKuttaExtrapolator::extrapolateNumericalJacobian( const LHCb::Magnet::MagneticFieldGrid* grid,
                                                                         const Tableau&                         table,
                                                                         RK::details::State<>& state, double zout,
                                                                         RK::details::Jacobian<>& jacobian ) const {
  // call the stanndard method but store the steps taken
  // FIXME: this is not guaranteed to produce the right number when running multithreaded...
  struct step_t {
    size_t numstep;
    size_t numfailedstep;
  };
  std::optional<step_t> cached{};
  if ( msgLevel( MSG::DEBUG ) ) {
    cached.emplace( m_totalstats.with_lock( []( const RK::Statistics& s ) {
      return step_t{ s.numstep, s.numfailedstep };
    } ) );
  }

  auto                inputstate = state;
  std::vector<double> stepvector;
  stepvector.reserve( 256 );
  RK::ErrorCode success =
      extrapolate( grid, table, state, zout, &jacobian, [&]( double step ) { stepvector.push_back( step ); } );
  if ( success == RK::ErrorCode::Success ) {
    // now make small changes in tx,ty,qop
    constexpr auto delta = std::array{ 0.01, 0.01, 1e-8 };
    for ( int col = 0; col < 3; ++col ) {
      auto astate = inputstate;
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
      RK::details::Vec4<>                  err;
      RK::details::Cache<Tableau::NStages> cache;
      for ( auto step : stepvector ) {
        if ( success =
                 evaluateStep( table, step, astate, err, cache,
                               [&]( const Gaudi::XYZPoint& position ) { return this->fieldVector( grid, position ); } );
             success != RK::ErrorCode::Success ) {
          return success;
        }
        ++cache.step;
      }
      if ( !( std::abs( state.z - astate.z ) < TrackParameters::propagationTolerance ) ) {
        m_numerical_integration_problem += 1;
        if ( cached ) {
          m_totalstats.with_lock( [&]( const RK::Statistics& s ) {
            // FIXME: this is not guaranteed to produce the right number when running multithreaded...
            std::cout << "problem in numerical integration.\n"
                      << "zin: " << inputstate.z << " "
                      << " zout: " << zout << " "
                      << " state.z: " << state.z << " "
                      << " dstate.z: " << astate.z << '\n'
                      << "num step: " << stepvector.size() << " " << s.numstep - cached->numstep << " "
                      << s.numfailedstep - cached->numfailedstep << std::endl;
          } );
        }
      }
      assert( std::abs( state.z - astate.z ) < TrackParameters::propagationTolerance );

      for ( int row = 0; row < 4; ++row ) {
        jacobian.matrix( row, col ) = ( astate.parameters( row ) - state.parameters( row ) ) / delta[col];
      }
    }
  }
  return success;
}
