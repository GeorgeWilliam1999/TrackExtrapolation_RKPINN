/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
\*****************************************************************************/
#pragma once

#include <cstdint>

/// Per-propagation sub-operation timing breakdown (RDTSC cycles)
struct ExtrapolatorSubTimers final {
  int64_t field_cycles    = 0; ///< Total cycles in fieldVector() calls
  int64_t deriv_cycles    = 0; ///< Total cycles in evaluateDerivatives()
  int64_t butcher_cycles  = 0; ///< Total cycles in Butcher stage accumulation
  int64_t jacobian_cycles = 0; ///< Total cycles in evaluateStepJacobian()
  int64_t stepsize_cycles = 0; ///< Total cycles in adaptive step-size control
  int64_t total_cycles    = 0; ///< Total propagation cycles
  int     nsteps          = 0; ///< Number of accepted RK steps
  int     nrejected       = 0; ///< Number of rejected steps
};
