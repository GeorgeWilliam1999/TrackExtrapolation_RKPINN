/*****************************************************************************\
* (c) Copyright 2000-2022 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#ifndef TRACKEXTRAPOLATORS_TRFIELDEXTRAPOLATORBASE_H
#define TRACKEXTRAPOLATORS_TRFIELDEXTRAPOLATORBASE_H

#include "TrackExtrapolator.h"
#include <Core/MagneticFieldGrid.h>
#include <DetDesc/GenericConditionAccessorHolder.h>
#include <GaudiKernel/DataObjectHandle.h>
#include <GaudiKernel/EventContext.h>
#include <GaudiKernel/ThreadLocalContext.h>
#include <Kernel/ILHCbMagnetSvc.h>
#include <Magnet/DeMagnet.h>
#include <optional>

/** @class TrackFieldExtrapolatorBase TrackFieldExtrapolatorBase.h TrackExtrapolators/TrackFieldExtrapolatorBase.h
 *
 *  A TrackFieldExtrapolatorBase is a TrackExtrapolator with access to the magnetic field
 *
 *  @author Wouter Hulsbergen
 *  @date   16/07/2009
 */

class TrackFieldExtrapolatorBase : public LHCb::DetDesc::ConditionAccessorHolder<TrackExtrapolator> {

public:
  using FieldVector   = Gaudi::XYZVector;
  using FieldGradient = Gaudi::Matrix3x3;
  /// constructor
  using ConditionAccessorHolder::ConditionAccessorHolder;

  /// initialize (picks up the field service)
  StatusCode initialize() override;

  /// access to the field
  FieldVector fieldVector( const LHCb::Magnet::MagneticFieldGrid* grid, const Gaudi::XYZPoint& position ) const {
    return m_fieldFunction( grid ? grid : currentGrid(), position );
  }

  /// access to the field gradient
  FieldGradient fieldGradient( const LHCb::Magnet::MagneticFieldGrid* grid, const Gaudi::XYZPoint& position ) const {
    return ( grid ? grid : currentGrid() )->fieldGradient( position );
  }

  bool usesGridInterpolation() const override { return m_useGridInterpolation; }

private:
  const LHCb::Magnet::MagneticFieldGrid* currentGrid() const {
    static thread_local struct {
      std::optional<EventContext::ContextID_t> evt;
      const LHCb::Magnet::MagneticFieldGrid*   grid{ nullptr };
    } current;
    auto& ctx = Gaudi::Hive::currentContext();
    if ( !current.evt || current.evt != ctx.evt() ) {
      current.evt  = ctx.evt();
      current.grid = m_magnet.get( getConditionContext( ctx ) ).fieldGrid();
    }
    assert( current.grid != nullptr );
    return current.grid;
  }

  Gaudi::XYZVector ( *m_fieldFunction )( const LHCb::Magnet::MagneticFieldGrid*, const Gaudi::XYZPoint& ) = nullptr;

  ConditionAccessor<DeMagnet> m_magnet{ this, "Magnet", LHCb::Det::Magnet::det_path };

  Gaudi::Property<bool> m_useGridInterpolation{ this, "UseGridInterpolation",
                                                true }; ///< Flag whether to interpolate on the grid or not
};

#endif // TRACKEXTRAPOLATORS_TRLINEAREXTRAPOLATOR_H
