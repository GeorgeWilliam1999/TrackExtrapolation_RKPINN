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
#include "DetDesc/IDetectorElement.h"
#include "DetDesc/Material.h"
#include "GaudiKernel/ServiceHandle.h"
#include "GaudiKernel/SystemOfUnits.h"
#include "MaterialLocatorBase.h"

/** @class DetailedMaterialLocator DetailedMaterialLocator.h
 *
 * Implementation of a IMaterialLocator that uses the TransportSvc for
 * finding materials on a trajectory.
 *
 *  @author Wouter Hulsbergen
 *  @date   21/05/2007
 */

class DetailedMaterialLocator : public MaterialLocatorBase {
public:
  /// Constructor
  using MaterialLocatorBase::intersect;
  using MaterialLocatorBase::MaterialLocatorBase;

protected:
  /// Intersect a line with volumes in the geometry
  ::Intersections intersect_volumes( const Gaudi::XYZPoint& p, const Gaudi::XYZVector& v, std::any& accelCache,
                                     IGeometryInfo const& geometry ) const override;

private:
  Gaudi::Property<double> m_minRadThickness{ this, "MinRadThickness", 1e-4 }; ///< minimum radiation thickness
  mutable Gaudi::Accumulators::MsgCounter<MSG::WARNING> m_outside{
      this, "No transport possible since destination is outside LHCb " };
};

DECLARE_COMPONENT( DetailedMaterialLocator )

namespace {
  inline const std::string chronotag = "DetailedMaterialLocator";
}

::Intersections DetailedMaterialLocator::intersect_volumes( const Gaudi::XYZPoint& start, const Gaudi::XYZVector& vect,
                                                            std::any&            accelCache,
                                                            IGeometryInfo const& geometry ) const {
  // check if transport is within LHCb
  constexpr double m_25m = 25 * Gaudi::Units::m;

  if ( std::abs( start.x() ) > m_25m || std::abs( start.y() ) > m_25m || std::abs( start.z() ) > m_25m ||
       std::abs( start.x() + vect.x() ) > m_25m || std::abs( start.y() + vect.y() ) > m_25m ||
       std::abs( start.z() + vect.z() ) > m_25m ) {
    ++m_outside;
    if ( msgLevel( MSG::DEBUG ) )
      debug() << "No transport between z= " << start.z() << " and " << start.z() + vect.z()
              << ", since it reaches outside LHCb"
              << "start = " << start << " vect= " << vect << endmsg;
    return {};
  } else {
    try {
      chronoSvc()->chronoStart( chronotag );
      const double mintick = 0;
      const double maxtick = 1;
      auto intersepts = m_tSvc->intersections( start, vect, mintick, maxtick, accelCache, geometry, m_minRadThickness );
      chronoSvc()->chronoStop( chronotag );
      return intersepts;
    } catch ( const GaudiException& exception ) {
      error() << "caught transportservice exception " << exception << '\n'
              << "propagating pos/vec: " << start << " / " << vect << endmsg;
      throw exception;
    }
  }
}

#ifdef USE_DD4HEP
struct SimplifiedMaterialLocator : public DetailedMaterialLocator {
  using DetailedMaterialLocator::DetailedMaterialLocator;
  StatusCode initialize() override {
    warning() << "In DD4HEP builds the SimplifiedMaterialLocator defaults to DetailedMaterialLocator because the "
                 "simplified geometry has not yet been implemented"
              << endmsg;
    return DetailedMaterialLocator::initialize();
  }
};
DECLARE_COMPONENT( SimplifiedMaterialLocator )
#endif
