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
#include "GaudiAlg/GaudiTool.h"
#include "GaudiKernel/SystemOfUnits.h"
#include "TrackInterfaces/ITrackExtraSelector.h"
#include "TrackInterfaces/ITrackExtrapolator.h"

/** @class TrackDistanceExtraSelector "TrackDistanceExtraSelector.h"
 *
 *  Distance selection of one extrapolator
 *
 */

class TrackDistanceExtraSelector : public extends<GaudiTool, ITrackExtraSelector> {

public:
  using extends::extends;

  const ITrackExtrapolator* select( double zStart, double zEnd ) const override {
    return std::abs( zEnd - zStart ) < m_shortDist ? m_shortDistanceExtrapolator.get()
                                                   : m_longDistanceExtrapolator.get();
  }

private:
  /// extrapolator to use for short transport in mag field
  ToolHandle<ITrackExtrapolator> m_shortDistanceExtrapolator{ this, "ShortDistanceExtrapolator",
                                                              "TrackParabolicExtrapolator/ShortDistanceExtrapolator" };
  /// extrapolator to use for long transport in mag field
  ToolHandle<ITrackExtrapolator> m_longDistanceExtrapolator{ this, "LongDistanceExtrapolator",
                                                             "TrackRungeKuttaExtrapolator/LongDistanceExtrapolator" };
  Gaudi::Property<double>        m_shortDist{ this, "shortDist", 100.0 * Gaudi::Units::mm };
};

DECLARE_COMPONENT( TrackDistanceExtraSelector )

//=============================================================================
