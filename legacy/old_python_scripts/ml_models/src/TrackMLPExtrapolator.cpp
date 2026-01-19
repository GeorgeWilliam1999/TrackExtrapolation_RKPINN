/*****************************************************************************\
* (c) Copyright 2000-2025 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

// STL
#include <array>
#include <cmath>
#include <fstream>
#include <vector>

// Gaudi
#include "GaudiKernel/PhysicalConstants.h"
#include "GaudiKernel/SystemOfUnits.h"

// Eigen for neural network operations
#include <Eigen/Dense>

// Local
#include "TrackFieldExtrapolatorBase.h"

// TrackEvent
#include "Event/TrackParameters.h"

// =============================================================================
// Simple Neural Network Implementation using Eigen
// =============================================================================
namespace {
  
  /// Activation functions
  inline Eigen::VectorXd relu( const Eigen::VectorXd& x ) {
    return x.cwiseMax( 0.0 );
  }
  
  inline Eigen::VectorXd tanh_act( const Eigen::VectorXd& x ) {
    return x.array().tanh();
  }
  
  inline Eigen::VectorXd sigmoid( const Eigen::VectorXd& x ) {
    return ( 1.0 / ( 1.0 + ( -x.array() ).exp() ) ).matrix();
  }

  /// Simple feedforward neural network
  class SimpleNN {
  public:
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
    std::string activation = "tanh";
    
    // Input/output normalization parameters
    Eigen::VectorXd input_mean;
    Eigen::VectorXd input_std;
    Eigen::VectorXd output_mean;
    Eigen::VectorXd output_std;
    
    bool isLoaded = false;
    
    /// Forward pass
    Eigen::VectorXd forward( const Eigen::VectorXd& input ) const {
      if ( !isLoaded ) {
        // Return input unchanged if no model loaded
        return input;
      }
      
      // Normalize input
      Eigen::VectorXd x = ( input.array() - input_mean.array() ) / input_std.array();
      
      // Hidden layers with activation
      for ( size_t i = 0; i < weights.size() - 1; ++i ) {
        x = weights[i] * x + biases[i];
        if ( activation == "relu" ) {
          x = relu( x );
        } else if ( activation == "tanh" ) {
          x = tanh_act( x );
        } else {
          x = sigmoid( x );
        }
      }
      
      // Output layer (linear)
      x = weights.back() * x + biases.back();
      
      // Denormalize output
      x = ( x.array() * output_std.array() + output_mean.array() ).matrix();
      
      return x;
    }
    
    /// Load model from binary file
    bool load( const std::string& filepath ) {
      std::ifstream file( filepath, std::ios::binary );
      if ( !file.is_open() ) {
        return false;
      }
      
      // Read number of layers
      int num_layers;
      file.read( reinterpret_cast<char*>( &num_layers ), sizeof( int ) );
      
      weights.resize( num_layers );
      biases.resize( num_layers );
      
      // Read each layer
      for ( int i = 0; i < num_layers; ++i ) {
        int rows, cols;
        file.read( reinterpret_cast<char*>( &rows ), sizeof( int ) );
        file.read( reinterpret_cast<char*>( &cols ), sizeof( int ) );
        
        weights[i].resize( rows, cols );
        biases[i].resize( rows );
        
        // Read weights
        file.read( reinterpret_cast<char*>( weights[i].data() ), rows * cols * sizeof( double ) );
        // Read biases
        file.read( reinterpret_cast<char*>( biases[i].data() ), rows * sizeof( double ) );
      }
      
      // Read normalization parameters
      int input_size, output_size;
      file.read( reinterpret_cast<char*>( &input_size ), sizeof( int ) );
      input_mean.resize( input_size );
      input_std.resize( input_size );
      file.read( reinterpret_cast<char*>( input_mean.data() ), input_size * sizeof( double ) );
      file.read( reinterpret_cast<char*>( input_std.data() ), input_size * sizeof( double ) );
      
      file.read( reinterpret_cast<char*>( &output_size ), sizeof( int ) );
      output_mean.resize( output_size );
      output_std.resize( output_size );
      file.read( reinterpret_cast<char*>( output_mean.data() ), output_size * sizeof( double ) );
      file.read( reinterpret_cast<char*>( output_std.data() ), output_size * sizeof( double ) );
      
      isLoaded = file.good();
      return isLoaded;
    }
    
    /// Initialize with random weights (for testing without trained model)
    void initializeRandom( const std::vector<int>& layer_sizes, int input_dim, int output_dim ) {
      weights.clear();
      biases.clear();
      
      int prev_size = input_dim;
      for ( int size : layer_sizes ) {
        // Xavier initialization
        double scale = std::sqrt( 2.0 / ( prev_size + size ) );
        weights.push_back( Eigen::MatrixXd::Random( size, prev_size ) * scale );
        biases.push_back( Eigen::VectorXd::Zero( size ) );
        prev_size = size;
      }
      // Output layer
      double scale = std::sqrt( 2.0 / ( prev_size + output_dim ) );
      weights.push_back( Eigen::MatrixXd::Random( output_dim, prev_size ) * scale );
      biases.push_back( Eigen::VectorXd::Zero( output_dim ) );
      
      // Default normalization (identity)
      input_mean = Eigen::VectorXd::Zero( input_dim );
      input_std = Eigen::VectorXd::Ones( input_dim );
      output_mean = Eigen::VectorXd::Zero( output_dim );
      output_std = Eigen::VectorXd::Ones( output_dim );
      
      isLoaded = true;
    }
  };
  
} // anonymous namespace

/** @class TrackMLPExtrapolator TrackMLPExtrapolator.cpp
 *
 *  A TrackMLPExtrapolator is a track extrapolator that uses a
 *  data-driven Multi-Layer Perceptron (MLP) neural network trained
 *  to approximate the Runge-Kutta equations of motion for charged
 *  particles in a magnetic field.
 *
 *  NOTE: This is a DATA-DRIVEN neural network, not a true Physics-Informed
 *  Neural Network (PINN). It learns from training data generated by the
 *  reference RK4 extrapolator, without explicit physics constraints in
 *  the loss function.
 *
 *  The MLP replaces the traditional adaptive RK integrator while
 *  maintaining the same interface.
 *
 *  @author G. Scriven
 *  @date   2025-12-19
 */
class TrackMLPExtrapolator : public TrackFieldExtrapolatorBase {

public:
  /// Constructor
  using TrackFieldExtrapolatorBase::TrackFieldExtrapolatorBase;

  /// Initialize the tool
  StatusCode initialize() override;

  /// Finalize the tool
  StatusCode finalize() override;

  using TrackFieldExtrapolatorBase::propagate;

  /// Propagate a state vector from zOld to zNew
  /// Transport matrix is calculated when transMat pointer is not NULL
  StatusCode propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew, Gaudi::TrackMatrix* transMat,
                        IGeometryInfo const& geometry, const LHCb::Tr::PID pid = LHCb::Tr::PID::Pion(),
                        const LHCb::Magnet::MagneticFieldGrid* grid = nullptr ) const override;

private:
  // ============================================================================
  // MLP-specific methods - neural network inference
  // ============================================================================

  /// Load the MLP model weights/configuration
  StatusCode loadModel();

  /// Run MLP inference to propagate state from z_in to z_out
  /// @param stateIn  Input state (x, y, tx, ty, qop) at z_in
  /// @param z_in     Initial z position
  /// @param z_out    Target z position
  /// @param stateOut Output state (x, y, tx, ty, qop) at z_out
  /// @return StatusCode indicating success or failure
  StatusCode propagateMLP( const std::array<double, 5>& stateIn, double z_in, double z_out,
                            std::array<double, 5>& stateOut ) const;

  /// Compute the transport Jacobian (5x5 matrix)
  /// This is critical for track fitting - must compute d(state_out)/d(state_in)
  /// @param stateIn  Input state at z_in
  /// @param z_in     Initial z position
  /// @param z_out    Target z position
  /// @param jacobian Output 5x5 Jacobian matrix
  /// @return StatusCode indicating success or failure
  StatusCode computeJacobian( const std::array<double, 5>& stateIn, double z_in, double z_out,
                              Gaudi::TrackMatrix& jacobian ) const;

  // ============================================================================
  // Tool properties
  // ============================================================================

  /// Path to the MLP model file (weights, architecture, etc.)
  Gaudi::Property<std::string> m_modelPath{ this, "ModelPath", "",
                                            "Path to the MLP model file" };

  /// Maximum allowed slope (protect against looping tracks)
  Gaudi::Property<double> m_maxSlope{ this, "MaxSlope", 10.,
                                      "Maximum allowed track slope (tx or ty)" };

  /// Maximum transverse position (protect against absurd tracks)
  Gaudi::Property<double> m_maxTransverse{ this, "MaxTransverse", 10. * Gaudi::Units::m,
                                           "Maximum allowed transverse position" };

  /// Use numerical Jacobian instead of analytical (for debugging/validation)
  Gaudi::Property<bool> m_numericalJacobian{ this, "NumericalJacobian", false,
                                             "Use numerical differentiation for Jacobian" };

  /// Step size for numerical Jacobian computation
  Gaudi::Property<double> m_jacobianDelta{ this, "JacobianDelta", 1e-5,
                                           "Step size for numerical Jacobian" };

  // ============================================================================
  // Counters for monitoring
  // ============================================================================

  mutable Gaudi::Accumulators::Counter<>                m_numCalls{ this, "NumCalls" };
  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_loopingTrack{
      this, "Looping track detected (large slope)", 10 };
  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_absurdTrack{
      this, "Absurd track detected (large transverse position)", 10 };
  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_mlpFailure{
      this, "MLP inference failed", 10 };

  // ============================================================================
  // MLP model storage
  // ============================================================================
  
  /// Neural network model
  mutable SimpleNN m_model;
  
  /// Hidden layer sizes for random initialization
  Gaudi::Property<std::vector<int>> m_hiddenLayers{ this, "HiddenLayers", {64, 64, 32},
                                                     "Hidden layer sizes for the neural network" };
  
  /// Activation function
  Gaudi::Property<std::string> m_activation{ this, "Activation", "tanh",
                                              "Activation function (relu, tanh, sigmoid)" };
};

// =============================================================================
// Implementation
// =============================================================================

DECLARE_COMPONENT( TrackMLPExtrapolator )

StatusCode TrackMLPExtrapolator::initialize() {
  return TrackFieldExtrapolatorBase::initialize().andThen( [&]() -> StatusCode {
    // Load the MLP model
    if ( !m_modelPath.value().empty() ) {
      auto sc = loadModel();
      if ( sc.isFailure() ) {
        error() << "Failed to load MLP model from: " << m_modelPath.value() << endmsg;
        return sc;
      }
      info() << "Loaded MLP model from: " << m_modelPath.value() << endmsg;
    } else {
      warning() << "No MLP model path specified - using placeholder implementation" << endmsg;
    }
    return StatusCode::SUCCESS;
  } );
}

StatusCode TrackMLPExtrapolator::finalize() {
  if ( msgLevel( MSG::DEBUG ) ) {
    debug() << "TrackMLPExtrapolator finalize - total calls: " << m_numCalls.sum() << endmsg;
  }
  return TrackFieldExtrapolatorBase::finalize();
}

StatusCode TrackMLPExtrapolator::loadModel() {
  if ( msgLevel( MSG::DEBUG ) ) {
    debug() << "Loading MLP model from: " << m_modelPath.value() << endmsg;
  }

  // Try to load model from file
  if ( !m_modelPath.value().empty() && m_model.load( m_modelPath.value() ) ) {
    info() << "Successfully loaded MLP model with " << m_model.weights.size() << " layers" << endmsg;
  } else {
    // Initialize with random weights for testing
    // Input: 6 features (x, y, tx, ty, qop, dz)
    // Output: 4 features (x_out, y_out, tx_out, ty_out)
    warning() << "Model file not found or invalid, initializing random network for testing" << endmsg;
    m_model.initializeRandom( m_hiddenLayers.value(), 6, 4 );
    m_model.activation = m_activation.value();
    
    // Set reasonable normalization for track parameters
    // Typical ranges: x,y ~ [-3000, 3000] mm, tx,ty ~ [-0.3, 0.3], qop ~ [-1e-4, 1e-4], dz ~ [0, 5000]
    m_model.input_mean = Eigen::VectorXd::Zero( 6 );
    m_model.input_std = ( Eigen::VectorXd( 6 ) << 1000.0, 1000.0, 0.2, 0.2, 1e-4, 2000.0 ).finished();
    m_model.output_mean = Eigen::VectorXd::Zero( 4 );
    m_model.output_std = ( Eigen::VectorXd( 4 ) << 1000.0, 1000.0, 0.2, 0.2 ).finished();
  }

  return StatusCode::SUCCESS;
}

StatusCode TrackMLPExtrapolator::propagate( Gaudi::TrackVector& stateVec, double zOld, double zNew,
                                                Gaudi::TrackMatrix* transMat, IGeometryInfo const& /*geometry*/,
                                                const LHCb::Tr::PID /*pid*/,
                                                const LHCb::Magnet::MagneticFieldGrid* /*grid*/ ) const {
  // Count calls
  ++m_numCalls;

  // Bail out if already at destination
  if ( std::abs( zOld - zNew ) < TrackParameters::propagationTolerance ) {
    if ( transMat ) *transMat = ROOT::Math::SMatrixIdentity();
    return StatusCode::SUCCESS;
  }

  // Extract state components
  // stateVec = (x, y, tx, ty, q/p)
  const double x   = stateVec( 0 );
  const double y   = stateVec( 1 );
  const double tx  = stateVec( 2 );
  const double ty  = stateVec( 3 );
  const double qop = stateVec( 4 );

  // Protect against absurd tracks
  if ( std::abs( x ) > m_maxTransverse || std::abs( y ) > m_maxTransverse ) {
    ++m_absurdTrack;
    if ( msgLevel( MSG::DEBUG ) ) {
      debug() << "Absurd track: x=" << x << ", y=" << y << " (max " << m_maxTransverse << ")" << endmsg;
    }
    return StatusCode::FAILURE;
  }

  // Protect against looping tracks
  if ( std::abs( tx ) > m_maxSlope || std::abs( ty ) > m_maxSlope ) {
    ++m_loopingTrack;
    if ( msgLevel( MSG::DEBUG ) ) {
      debug() << "Looping track: tx=" << tx << ", ty=" << ty << " (max " << m_maxSlope << ")" << endmsg;
    }
    return StatusCode::FAILURE;
  }

  // Prepare input state array
  std::array<double, 5> stateIn  = { x, y, tx, ty, qop };
  std::array<double, 5> stateOut = {};

  // Run MLP inference
  auto sc = propagateMLP( stateIn, zOld, zNew, stateOut );
  if ( sc.isFailure() ) {
    ++m_mlpFailure;
    return sc;
  }

  // Update state vector with propagated values
  stateVec( 0 ) = stateOut[0]; // x
  stateVec( 1 ) = stateOut[1]; // y
  stateVec( 2 ) = stateOut[2]; // tx
  stateVec( 3 ) = stateOut[3]; // ty
  stateVec( 4 ) = stateOut[4]; // q/p (typically unchanged)

  // Compute transport matrix if requested
  if ( transMat ) {
    sc = computeJacobian( stateIn, zOld, zNew, *transMat );
    if ( sc.isFailure() ) {
      return sc;
    }
  }

  return StatusCode::SUCCESS;
}

StatusCode TrackMLPExtrapolator::propagateMLP( const std::array<double, 5>& stateIn, double z_in, double z_out,
                                                    std::array<double, 5>& stateOut ) const {
  // ==========================================================================
  // Neural Network Inference for Track Propagation
  // ==========================================================================
  //
  // Input features:
  //   - x, y, tx, ty, qop: track state at z_in
  //   - dz: propagation distance (z_out - z_in)
  //
  // Output:
  //   - x, y, tx, ty at z_out (qop unchanged)
  //
  // ==========================================================================

  const double dz = z_out - z_in;
  
  // Prepare input vector: [x, y, tx, ty, qop, dz]
  Eigen::VectorXd input( 6 );
  input << stateIn[0], stateIn[1], stateIn[2], stateIn[3], stateIn[4], dz;
  
  // Run neural network inference
  Eigen::VectorXd output = m_model.forward( input );
  
  // If model is not loaded, fall back to linear extrapolation
  if ( !m_model.isLoaded ) {
    stateOut[0] = stateIn[0] + stateIn[2] * dz;  // x + tx * dz
    stateOut[1] = stateIn[1] + stateIn[3] * dz;  // y + ty * dz
    stateOut[2] = stateIn[2];                     // tx unchanged
    stateOut[3] = stateIn[3];                     // ty unchanged
    stateOut[4] = stateIn[4];                     // q/p unchanged
  } else {
    // Use neural network output
    stateOut[0] = output( 0 );  // x
    stateOut[1] = output( 1 );  // y
    stateOut[2] = output( 2 );  // tx
    stateOut[3] = output( 3 );  // ty
    stateOut[4] = stateIn[4];   // q/p (unchanged, not predicted by NN)
  }

  if ( msgLevel( MSG::VERBOSE ) ) {
    verbose() << "MLP propagate: z " << z_in << " -> " << z_out
              << " state: (" << stateIn[0] << ", " << stateIn[1] << ", "
              << stateIn[2] << ", " << stateIn[3] << ", " << stateIn[4] << ")"
              << " -> (" << stateOut[0] << ", " << stateOut[1] << ", "
              << stateOut[2] << ", " << stateOut[3] << ", " << stateOut[4] << ")" << endmsg;
  }

  return StatusCode::SUCCESS;
}

StatusCode TrackMLPExtrapolator::computeJacobian( const std::array<double, 5>& stateIn, double z_in, double z_out,
                                                      Gaudi::TrackMatrix& jacobian ) const {
  // ==========================================================================
  // Compute the 5x5 transport Jacobian: d(state_out) / d(state_in)
  // ==========================================================================
  //
  // The Jacobian matrix structure:
  //
  //       (  dx/dx0    dx/dy0    dx/dtx0    dx/dty0    dx/dqop0   )
  //       (  dy/dx0    dy/dy0    dy/dtx0    dy/dty0    dy/dqop0   )
  //   J = ( dtx/dx0   dtx/dy0   dtx/dtx0   dtx/dty0   dtx/dqop0   )
  //       ( dty/dx0   dty/dy0   dty/dtx0   dty/dty0   dty/dqop0   )
  //       ( dqop/dx0  dqop/dy0  dqop/dtx0  dqop/dty0  dqop/dqop0  )
  //
  // Options for computing this:
  // 1. Analytical derivatives from MLP (if available)
  // 2. Automatic differentiation (PyTorch autograd, JAX, etc.)
  // 3. Numerical differentiation (finite differences)
  //
  // ==========================================================================

  if ( m_numericalJacobian ) {
    // Numerical Jacobian via finite differences
    const double delta = m_jacobianDelta;

    for ( int col = 0; col < 5; ++col ) {
      // Perturb state in positive direction
      std::array<double, 5> statePlus = stateIn;
      statePlus[col] += delta;
      std::array<double, 5> outPlus;
      auto sc = propagateMLP( statePlus, z_in, z_out, outPlus );
      if ( sc.isFailure() ) return sc;

      // Perturb state in negative direction
      std::array<double, 5> stateMinus = stateIn;
      stateMinus[col] -= delta;
      std::array<double, 5> outMinus;
      sc = propagateMLP( stateMinus, z_in, z_out, outMinus );
      if ( sc.isFailure() ) return sc;

      // Central difference
      for ( int row = 0; row < 5; ++row ) {
        jacobian( row, col ) = ( outPlus[row] - outMinus[row] ) / ( 2.0 * delta );
      }
    }
  } else {
    // ==========================================================================
    // TODO: IMPLEMENT ANALYTICAL OR AUTODIFF JACOBIAN HERE
    // ==========================================================================
    //
    // If your MLP framework supports automatic differentiation:
    //   jacobian = torch::autograd::jacobian(pinn_forward, input_state);
    //
    // Or if you have analytical derivatives from your PINN architecture.
    //
    // ==========================================================================

    // PLACEHOLDER: Identity-like Jacobian for linear extrapolation
    const double dz = z_out - z_in;

    jacobian = ROOT::Math::SMatrixIdentity();
    // For linear extrapolation: x_out = x_in + tx * dz
    jacobian( 0, 2 ) = dz;  // dx/dtx = dz
    jacobian( 1, 3 ) = dz;  // dy/dty = dz
    // tx, ty, qop are unchanged in placeholder, so diagonal = 1
  }

  return StatusCode::SUCCESS;
}
