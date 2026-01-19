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
#include "TrackInterfaces/ITrackExtraSelector.h"
#include "TrackInterfaces/ITrackExtrapolator.h"
#include <string>

/** @class TrackSimpleExtraSelector "TrackSimpleExtraSelector.h"
 *
 *  Simple selection of one extrapolator
 *
 */

class TrackSimpleExtraSelector : public extends<GaudiTool, ITrackExtraSelector> {

public:
  using extends::extends;

  const ITrackExtrapolator* select( double, double ) const override { return m_extrapolator.get(); }

private:
  ToolHandle<ITrackExtrapolator> m_extrapolator{ this, "ExtrapolatorName", "TrackParabolicExtrapolator" };
};

DECLARE_COMPONENT( TrackSimpleExtraSelector )
