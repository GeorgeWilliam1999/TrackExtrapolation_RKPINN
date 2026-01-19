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

#pragma once

// Include files
// -------------
// from Gaudi
#include "GaudiAlg/GaudiTool.h"
#include "GaudiKernel/SystemOfUnits.h"
#include "GaudiKernel/ToolHandle.h"

// from TrackInterfaces
#include "TrackInterfaces/IMaterialLocator.h"
#include "TrackInterfaces/IStateCorrectionTool.h"

#include "DetDesc/ITransportSvc.h"

/** @class MaterialLocatorBase MaterialLocatorBase.h
 *
 *  A MaterialLocatorBase is a base class implementing methods
 *  from the IMaterialLocatorBase interface.
 *
 *  @author Wouter Hulsbergen
 *  @date   12/05/2007
 */

class MaterialLocatorBase : public extends<GaudiTool, IMaterialLocator> {
public:
  /// Constructor
  using extends::extends;

  using IMaterialLocator::intersect;

  // Create an instance of the accelerator cache
  std::any createCache() const override { return m_tSvc->createCache(); }

  /// Intersect a trajectory with volumes in the geometry
  Intersections intersect( const LHCb::ZTrajectory<double>& traj, std::any& accelCache,
                           IGeometryInfo const& geometry ) const override;

  void applyMaterialCorrections( LHCb::State& stateAtTarget, const IMaterialLocator::Intersections& intersepts,
                                 double zorigin, const LHCb::Tr::PID pid, bool applyScatteringCorrection,
                                 bool applyEnergyLossCorrection ) const override;

protected:
  /// Intersect a line with volumes in the geometry
  virtual ::Intersections intersect_volumes( const Gaudi::XYZPoint& p, const Gaudi::XYZVector& v, std::any& accelCache,
                                             IGeometryInfo const& geometry ) const = 0;

private:
  /// Intersect a line with volumes in the geometry
  Intersections intersect_point( const Gaudi::XYZPoint& p, const Gaudi::XYZVector& v, std::any& accelCache,
                                 IGeometryInfo const& geometry ) const;

  static constexpr size_t m_maxNumIntervals             = 20;
  static constexpr double m_maxDeviation                = 5 * Gaudi::Units::cm;
  static constexpr double m_maxDeviationAtRefstates     = 2 * Gaudi::Units::mm;
  static constexpr double m_maxDeviationAtVeloRefstates = 0.5 * Gaudi::Units::mm;

  ToolHandle<IStateCorrectionTool> m_scatteringTool{ this, "StateMSCorrectionTool", "StateThickMSCorrectionTool" };
  ToolHandle<IStateCorrectionTool> m_dedxtool{ this, "GeneralDedxTool", "StateDetailedBetheBlochEnergyCorrectionTool" };
  ToolHandle<IStateCorrectionTool> m_elecdedxtool{ this, "ElectronDedxTool", "StateElectronEnergyCorrectionTool" };

protected:
  /// Transport service
  ServiceHandle<ITransportSvc> m_tSvc{
#ifdef USE_DD4HEP
      "TGeoTransportSvc",
#else
      "TransportSvc",
#endif
      name() };
};
