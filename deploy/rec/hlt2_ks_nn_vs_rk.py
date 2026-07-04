###############################################################################
# KS0 (LL + DD) reconstruction and vertex fitting with the state fetches
# served either by the production extrapolator stack (default) or with the
# neural vertex-fit surrogate as TrackMasterExtrapolator's field engine
# (environment VF_NN_EXTRAP=1). Material handling (multiple scattering,
# energy loss) stays with TrackMasterExtrapolator in BOTH variants — only the
# through-field propagation engine differs, which is the production-faithful
# wiring for the surrogate (NNVertexFitExtrapolator, Rec!Tr/TrackExtrapolators).
#
# Run (from the stack root, with the matching input+conds options first):
#   ./Moore/run gaudirun.py Moore/Hlt/Moore/tests/options/mdf_input_and_conds_hlt2.py \
#       Moore/Hlt/RecoConf/options/hlt2_ks_nn_vs_rk.py
#   VF_NN_EXTRAP=1 ./Moore/run gaudirun.py ... (same files) for the NN variant.
#
# Compare: KS0 combiner counters, monitor output, and the NN tool's
# served/fallback counters in the finalize summary.
###############################################################################
import os

from RecoConf.config import Reconstruction, run_reconstruction
from RecoConf.event_filters import require_gec, require_pvs
from RecoConf.global_tools import stateProvider_with_simplified_geom
from RecoConf.options import options
from RecoConf.reconstruction_objects import make_pvs, reconstruction
from RecoConf.standard_particles import make_KsDD, make_KsLL

USE_NN = os.environ.get("VF_NN_EXTRAP", "0") == "1"
options.evt_max = int(os.environ.get("VF_EVT_MAX", "300"))


def stateProvider_with_nn_field_engine():
    """TrackStateProvider whose TrackMasterExtrapolator delegates the
    through-field steps to the neural surrogate (RK fallback inside the tool
    for out-of-domain requests); mirrors global_tools.py otherwise."""
    from PyConf.Tools import (
        NNVertexFitExtrapolator,
        SimplifiedMaterialLocator,
        TrackInterpolator,
        TrackMasterExtrapolator,
        TrackSimpleExtraSelector,
        TrackStateProvider,
    )

    nn = NNVertexFitExtrapolator(public=True)
    tme = TrackMasterExtrapolator(
        public=True,
        MaterialLocator=SimplifiedMaterialLocator(),
        ExtraSelector=TrackSimpleExtraSelector(ExtrapolatorName=nn),
    )
    return TrackStateProvider(
        public=True, Extrapolator=tme, Interpolator=TrackInterpolator(Extrapolator=tme)
    )


def ks_reconstruction():
    ksll = make_KsLL()
    ksdd = make_KsDD()
    prefilters = [require_gec(), require_pvs(make_pvs())]
    return Reconstruction("ks_nn_vs_rk", [ksll, ksdd], prefilters)


public_tools = (
    [stateProvider_with_nn_field_engine()]
    if USE_NN
    else [stateProvider_with_simplified_geom()]
)

print("=" * 70)
print(f"KS reco variant: {'NN field engine' if USE_NN else 'production default'}")
print("=" * 70)

with reconstruction.bind(from_file=False, spruce=False):
    run_reconstruction(options, ks_reconstruction, public_tools)
