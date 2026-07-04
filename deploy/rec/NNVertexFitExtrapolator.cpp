/*****************************************************************************\
* (c) Copyright 2000-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "Event/TrackParameters.h"
#include "GaudiKernel/ToolHandle.h"
#include "TrackExtrapolator.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

/** @class NNVertexFitExtrapolator NNVertexFitExtrapolator.cpp
 *
 *  Neural surrogate for the through-field state fetch of the vertex fitters
 *  (TrackStateProvider -> DecayTreeFitter / ParticleVertexFitter).
 *
 *  The model is the vertex-fit PINN_v2 surrogate (track-extrapolation-pinn
 *  repo, experiment vf_zfeat_jacrow_h96): an 8-feature, two-layer tanh
 *  encoder whose 4 outputs enter through a kick-scaled head,
 *      out = straight_line(state, dz) + g * (kappa*qhat*dz) * corr,
 *  so the exact 1/p scaling of the magnetic bend is built in. The transport
 *  matrix uses the head-only Jacobian: the straight-line block plus the
 *  EXACT momentum column d(out)/d(q/p) = correction/(q/p), which the head's
 *  linearity in q/p provides at no extra cost (validated at fit level:
 *  gates/run_vf_p4_fitharness.py, NNH arm).
 *
 *  Units: the network was trained in the corpus convention
 *  qhat = c_light[mm/ns] * (q/p)[1/MeV]; this tool converts from and to the
 *  Gaudi q/p on the fly (incl. the Jacobian momentum column).
 *
 *  Requests outside the training domain (the four vertex-fit leg classes,
 *  MagDown, p in [1.5, 120] GeV) are delegated to a fallback extrapolator
 *  (default: TrackRungeKuttaExtrapolator). |dz| < 5 mm is transported as a
 *  straight line, which is exact to sub-micrometre there and matches the
 *  corpus contract (legs below 5 mm were excluded because consumers
 *  transport linearly anyway).
 *
 *  Weights: flat fp32 "VFK1" blob written by deploy/vf_export_weights.py.
 *
 *  @author George Scriven (track-extrapolation-pinn vertex-fit line)
 *  @date   2026-07-04
 */

namespace {

  constexpr float  kKappa    = 1.0e-3f;      // corpus EOM prefactor
  constexpr double kQopScale = 299.792458;   // qhat = kQopScale * (q/p)[1/MeV]

  struct VfKernel {
    int                H = 0;
    std::array<float, 7> mean{}, std{};
    std::vector<float>   W1, b1, W2, b2, W3, b3, g4;

    bool load( const std::string& path ) {
      std::ifstream f( path, std::ios::binary );
      if ( !f.is_open() ) return false;
      char magic[4];
      f.read( magic, 4 );
      if ( !f.good() || std::memcmp( magic, "VFK1", 4 ) != 0 ) return false;
      std::int32_t h = 0;
      f.read( reinterpret_cast<char*>( &h ), sizeof( h ) );
      if ( !f.good() || h <= 0 || h > 4096 ) return false;
      H          = h;
      auto block = [&f]( float* dst, size_t n ) {
        f.read( reinterpret_cast<char*>( dst ), n * sizeof( float ) );
        return f.good();
      };
      auto vblock = [&f]( std::vector<float>& v, size_t n ) {
        v.resize( n );
        f.read( reinterpret_cast<char*>( v.data() ), n * sizeof( float ) );
        return f.good();
      };
      return block( mean.data(), 7 ) && block( std.data(), 7 ) &&
             vblock( W1, size_t( H ) * 8 ) && vblock( b1, H ) &&
             vblock( W2, size_t( H ) * H ) && vblock( b2, H ) &&
             vblock( W3, size_t( 4 ) * H ) && vblock( b3, 4 ) && vblock( g4, 4 );
    }

    /// forward pass; in = (x, y, tx, ty, qhat, z0, dz); out4 = corrections
    /// applied on top of the straight line by the caller via c[]
    void corrections( const double* in7, float c[4] ) const {
      float e[8];
      for ( int k = 0; k < 5; ++k ) e[k] = ( float( in7[k] ) - mean[k] ) / std[k];
      e[5] = 1.0f; // legacy envelope slot (z_frac = 1 in single-shot deployment)
      e[6] = ( float( in7[5] ) - mean[5] ) / std[5];
      e[7] = ( float( in7[6] ) - mean[6] ) / std[6];

      std::vector<float> h1( H ), h2( H );
      for ( int i = 0; i < H; ++i ) {
        const float* w = &W1[size_t( i ) * 8];
        float        a = b1[i];
        for ( int k = 0; k < 8; ++k ) a += w[k] * e[k];
        h1[i] = std::tanh( a );
      }
      for ( int i = 0; i < H; ++i ) {
        const float* w = &W2[size_t( i ) * H];
        float        a = b2[i];
        for ( int k = 0; k < H; ++k ) a += w[k] * h1[k];
        h2[i] = std::tanh( a );
      }
      for ( int i = 0; i < 4; ++i ) {
        const float* w = &W3[size_t( i ) * H];
        float        a = b3[i];
        for ( int k = 0; k < H; ++k ) a += w[k] * h2[k];
        c[i] = a;
      }
    }
  };

  /// the training-domain test: the four vertex-fit leg classes (MagDown).
  /// Besides the (z0, z1, p) boxes, legs A/C/D require POINTING CONSISTENCY:
  /// the straight-line position at the target plane must lie near the beam
  /// axis, as every vertex-fit fetch does (corpus maximum 76 mm; cut 100 mm).
  /// The corpus is sampled on that physical manifold, and off it the network
  /// extrapolates poorly (found by the uniform-grid in-stack test, where
  /// x = tx*z rows point up to 360 mm off-axis at the target and the model
  /// errs at mm scale while being at spec on-manifold). Leg B is exempt:
  /// there the straight-line miss IS the magnet bend (up to ~3.6 m).
  inline bool inDomain( const Gaudi::TrackVector& s, double z0, double z1, double absQhat ) {
    if ( absQhat < 0.0025 || absQhat > 0.20 ) return false; // p outside [1.5, 120] GeV
    const double dz   = z1 - z0;
    const bool   legB = ( z0 >= 7500. && z0 <= 9410. && z1 >= 0. && z1 <= 2300. );
    if ( legB ) return true;
    const bool legA = ( z0 >= 2300. && z0 <= 2700. && z1 >= 0. && z1 <= 2300. );
    const bool legC = ( z0 >= 0. && z0 <= 770. && z1 >= -200. && z1 <= 800. );
    const bool legD = ( std::abs( dz ) <= 500. && z0 >= 0. && z0 <= 2800. && z1 >= 0. && z1 <= 2800. );
    if ( !( legA || legC || legD ) ) return false;
    const double xAt = s[0] + s[2] * dz;
    const double yAt = s[1] + s[3] * dz;
    return ( xAt * xAt + yAt * yAt ) < 100. * 100.;
  }

} // namespace

struct NNVertexFitExtrapolator : TrackExtrapolator {

  using TrackExtrapolator::TrackExtrapolator;
  using TrackExtrapolator::propagate;

  StatusCode initialize() override {
    return TrackExtrapolator::initialize().andThen( [&]() -> StatusCode {
      if ( !m_kernel.load( m_weightsFile.value() ) ) {
        error() << "failed to load VFK1 weights blob '" << m_weightsFile.value() << "'" << endmsg;
        return StatusCode::FAILURE;
      }
      info() << "loaded vertex-fit surrogate weights (H=" << m_kernel.H << ") from " << m_weightsFile.value()
             << endmsg;
      return m_fallback.retrieve();
    } );
  }

  /// Propagate a state vector from zOld to zNew; transport matrix (head-only
  /// Jacobian) is calculated when transMat is not null.
  StatusCode propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew, Gaudi::TrackMatrix* transMat,
                        IGeometryInfo const& geometry, const LHCb::Tr::PID pid = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

private:
  Gaudi::Property<std::string> m_weightsFile{ this, "WeightsFile",
                                              "/data/bfys/gscriven/TrackExtrapolation/experiments/vertexfit/results/"
                                              "vf_kernel_vf_zfeat_jacrow_h96.blob" };
  ToolHandle<ITrackExtrapolator> m_fallback{ this, "FallbackExtrapolator", "TrackRungeKuttaExtrapolator" };

  VfKernel m_kernel;

  mutable Gaudi::Accumulators::Counter<> m_nNN{ this, "#calls served by the surrogate" };
  mutable Gaudi::Accumulators::Counter<> m_nStraight{ this, "#calls served as straight line (|dz| < 5 mm)" };
  mutable Gaudi::Accumulators::Counter<> m_nFallback{ this, "#calls delegated to the fallback extrapolator" };
};

DECLARE_COMPONENT( NNVertexFitExtrapolator )

StatusCode NNVertexFitExtrapolator::propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew,
                                               Gaudi::TrackMatrix* transMat, IGeometryInfo const& geometry,
                                               const LHCb::Tr::PID pid, const LHCb::Magnet::MagneticFieldGrid* grid ) const {
  const double dz = zNew - zOld;

  // already there
  if ( std::abs( dz ) < TrackParameters::propagationTolerance ) {
    if ( transMat ) *transMat = ROOT::Math::SMatrixIdentity();
    return StatusCode::SUCCESS;
  }

  // below the corpus minimum leg the straight line is exact to sub-um
  if ( std::abs( dz ) < 5.0 ) {
    ++m_nStraight;
    if ( transMat ) {
      *transMat           = ROOT::Math::SMatrixIdentity();
      ( *transMat )( 0, 2 ) = dz;
      ( *transMat )( 1, 3 ) = dz;
    }
    stateVec[0] += stateVec[2] * dz;
    stateVec[1] += stateVec[3] * dz;
    return StatusCode::SUCCESS;
  }

  const double qhat = stateVec[4] * kQopScale;
  if ( !inDomain( stateVec, zOld, zNew, std::abs( qhat ) ) ) {
    ++m_nFallback;
    return m_fallback->propagate( stateVec, zOld, zNew, transMat, geometry, pid, grid );
  }

  ++m_nNN;
  const double in7[7] = { stateVec[0], stateVec[1], stateVec[2], stateVec[3], qhat, zOld, dz };
  float        c[4];
  m_kernel.corrections( in7, c );

  // kick-scaled head (fp32 like the training/deployment gates)
  const float kd   = kKappa * float( qhat ) * float( dz );
  const float dTx  = m_kernel.g4[0] * kd * c[0];
  const float dTy  = m_kernel.g4[1] * kd * c[1];
  const float dX   = m_kernel.g4[2] * kd * float( dz ) * c[2];
  const float dY   = m_kernel.g4[3] * kd * float( dz ) * c[3];

  if ( transMat ) {
    // head-only Jacobian: straight-line block + the exact momentum column
    // d(out)/d(qop_gaudi) = (correction / qhat) * kQopScale
    *transMat           = ROOT::Math::SMatrixIdentity();
    ( *transMat )( 0, 2 ) = dz;
    ( *transMat )( 1, 3 ) = dz;
    const double col = kQopScale / qhat;
    ( *transMat )( 0, 4 ) = double( dX ) * col;
    ( *transMat )( 1, 4 ) = double( dY ) * col;
    ( *transMat )( 2, 4 ) = double( dTx ) * col;
    ( *transMat )( 3, 4 ) = double( dTy ) * col;
  }

  stateVec[0] += stateVec[2] * dz + double( dX );
  stateVec[1] += stateVec[3] * dz + double( dY );
  stateVec[2] += double( dTx );
  stateVec[3] += double( dTy );
  // stateVec[4] (q/p) unchanged by field-only transport

  return StatusCode::SUCCESS;
}
