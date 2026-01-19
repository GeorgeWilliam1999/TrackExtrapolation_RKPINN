#pragma once

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

#include <Eigen/Dense>

//
// a few functors to make a basis of functions for parametrizations.
//
// author: Wouter Hulsbergen, September 2016
//

namespace BasisFunctions {

  // create a multi dimensional function basis as the product of lower
  // dimensional basis functions.
  template <typename BasicFunctionA, typename BasicFunctionB>
  struct BasicFunctionProduct {
    enum {
      NumArgs   = static_cast<int>( BasicFunctionA::NumArgs ) + static_cast<int>( BasicFunctionB::NumArgs ),
      NumValues = static_cast<int>( BasicFunctionA::NumValues ) * static_cast<int>( BasicFunctionB::NumValues )
    };
    typedef typename BasicFunctionA::ArgType             ArgType;
    typedef typename BasicFunctionA::ValueType           ValueType;
    typedef Eigen::Matrix<ArgType, NumArgs, 1>           Arguments;
    typedef Eigen::Matrix<ValueType, NumValues, 1>       Value;
    typedef Eigen::Matrix<ValueType, NumValues, NumArgs> Derivative;

    Value evaluate( const typename BasicFunctionA::ArgType* pars ) const {
      // this should have worked, but if it does not know the value of
      // the integers, it does not compile :-( I think that hat is
      // because the 'head', 'tail' and 'block' functions depend on what
      // value the integer has. I cannot get themn to work, so we'll
      // just deal with pointers :-(
      // const size_t NA = BasicFunctionA::NumArgs ;
      // const size_t NB = BasicFunctionB::NumArgs ;
      // auto valA = BasicFunctionA().evaluate( pars.head<NA>() ) ;
      // auto valB = BasicFunctionB().evaluate( pars.tail<NB>() ) ;
      Value rc;
      auto  valA = BasicFunctionA().evaluate( pars );
      auto  valB = BasicFunctionB().evaluate( pars + BasicFunctionA::NumArgs );
      for ( size_t j = 0; j < BasicFunctionB::NumValues; ++j )
        for ( size_t i = 0; i < BasicFunctionA::NumValues; ++i )
          rc[i + BasicFunctionA::NumValues * j] = valA[i] * valB[j];
      return rc;
    }

    Derivative evaluateDerivative( const typename BasicFunctionA::ArgType* pars ) const {
      auto       valA = BasicFunctionA().evaluate( pars );
      auto       valB = BasicFunctionB().evaluate( pars + BasicFunctionA::NumArgs );
      auto       derA = BasicFunctionA().evaluateDerivative( pars );
      auto       derB = BasicFunctionB().evaluateDerivative( pars + BasicFunctionA::NumArgs );
      Derivative rc;
      for ( size_t j = 0; j < BasicFunctionB::NumValues; ++j )
        for ( size_t i = 0; i < BasicFunctionA::NumValues; ++i ) {
          for ( size_t iarg = 0; iarg < BasicFunctionA::NumArgs; ++iarg )
            rc( i + BasicFunctionA::NumValues * j, iarg ) = derA( i, iarg ) * valB( j );
          for ( size_t iarg = 0; iarg < BasicFunctionB::NumArgs; ++iarg )
            rc( i + BasicFunctionA::NumValues * j, iarg + BasicFunctionA::NumArgs ) = valA( i ) * derB( j, iarg );
        }
      return rc;
    }

    Value      evaluate( const Arguments& pars ) const { return evaluate( &( pars( 0, 0 ) ) ); }
    Derivative evaluateDerivative( const Arguments& pars ) const { return evaluateDerivative( &( pars( 0, 0 ) ) ); }
  };

  // Create a set of basis functions that are 'standard' polynomials of order 0 ... N
  template <size_t Order, typename ArgTypeT = double, typename ValueTypeT = double>
  struct Polynomial1D {
    enum { NumArgs = 1, NumValues = Order + 1 };
    typedef ArgTypeT                               ArgType;
    typedef ValueTypeT                             ValueType;
    typedef Eigen::Matrix<ArgType, NumArgs, 1>     Arguments;
    typedef Eigen::Matrix<ValueType, NumValues, 1> Value;
    typedef Value                                  Derivative;

    Value evaluate( const ArgTypeT x ) const {
      Value rc;
      rc[0] = 1;
      for ( size_t i = 1; i <= Order; ++i ) rc[i] = x * rc[i - 1];
      return rc;
    }
    Derivative evaluateDerivative( const ArgTypeT x ) const {
      Derivative rc;
      rc[0]   = 0;
      Value y = evaluate( x );
      for ( size_t i = 1; i <= Order; ++i ) rc[i] = i * y[i - 1];
      return rc;
    }
    auto evaluate( const ArgTypeT* x ) const { return evaluate( *x ); }
    auto evaluateDerivative( const ArgTypeT* x ) const { return evaluateDerivative( *x ); }
    auto evaluate( const Arguments& x ) const { return evaluate( x( 0, 0 ) ); }
    auto evaluateDerivative( const Arguments& x ) const { return evaluateDerivative( x( 0, 0 ) ); }
  };

  // Create a set of basis functions that are Chebychev's of order 0 ... N
  template <size_t Order, typename ArgTypeT = double, typename ValueTypeT = double>
  struct ChebychevPolynomial1D {
    enum { NumArgs = 1, NumValues = Order + 1 };
    typedef ArgTypeT                               ArgType;
    typedef ValueTypeT                             ValueType;
    typedef Eigen::Matrix<ArgType, NumArgs, 1>     Arguments;
    typedef Eigen::Matrix<ValueType, NumValues, 1> Value;
    // typedef std::array<ValueTypeT,Order+1> Value ;
    // typedef Eigen::Matrix<ValueTypeT,Order+1,1> Value ;
    typedef Value Derivative;

    Value evaluate( const ArgTypeT x ) const {
      Value rc;
      rc[0] = 1;
      // it would be nicer to do this by template specialization
      if ( Order >= 1 ) {
        rc[1] = x;
        for ( size_t i = 2; i <= Order; ++i ) rc[i] = 2 * x * rc[i - 1] - rc[i - 2];
      }
      return rc;
    }
    Derivative evaluateDerivative( const ArgTypeT x ) const {
      Derivative rc;
      rc[0] = 0;
      // it would be nicer to do this by template specialization
      if ( Order >= 1 ) {
        rc[1] = 1;
        if ( Order >= 2 ) {
          Value y = evaluate( x );
          for ( size_t i = 2; i <= Order; ++i ) rc[i] = 2 * y[i - 1] + 2 * x * rc[i - 1] - rc[i - 2];
        }
      }
      return rc;
    }

    auto evaluate( const ArgTypeT* x ) const { return evaluate( *x ); }
    auto evaluateDerivative( const ArgTypeT* x ) const { return evaluateDerivative( *x ); }
    auto evaluate( const Arguments& x ) const { return evaluate( x( 0, 0 ) ); }
    auto evaluateDerivative( const Arguments& x ) const { return evaluateDerivative( x( 0, 0 ) ); }
  };
} // namespace BasisFunctions
