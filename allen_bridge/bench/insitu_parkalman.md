# Tier-2 — in-situ ParKalman reality check (recipe)

Tier-2 cross-checks the isolated Tier-1 micro-bench against the **real ParKalman
code path**. Two routes are documented; the first is the one **executed** here, the
second is the gold-standard GPU sequence-monitor recipe for a CUDA Allen build.

## What `m_use_nn_utt` actually does (read the code first)

`ExtrapolateUTT` (`ParKalmanMethods.cuh:486`) is a **hybrid**:

```
extrapUTT(...)               // ALWAYS runs: state + Jacobian F + noise Q
if (use_nn_utt) {
    pinn_v2_utt_state(...)    // ADDS a forward pass; overwrites x[0..3] only
}
```

So switching `use_nn_utt` **on does not remove** the polynomial — it keeps
extrapUTT for the covariance transport and *adds* one PINN forward pass. The
in-situ "stock → NN" delta is therefore **+1 PINN pass**, not a saving. The
quantity that must agree with Tier-1 (within 2×) is the **extrapUTT : PINN
per-track cost ratio**, which is platform-invariant (a ratio of two device
functions timed on the same hardware), so a CPU measurement is a valid cross-check
of the GPU ratio.

## Route A — executed CPU in-situ harness (`insitu_parkalman.cpp`)

The built Allen is a **CPU target** (`CMakeCache.txt: TARGET_DEVICE=CPU`), so a
faithful in-situ timing is a CPU build of the verbatim Allen device functions over
the **same gen-4 tracks** as Tier-1:

```bash
ALLEN_DIR=/data/bfys/gscriven/Allen STACK_DIR=/data/bfys/gscriven/TE_stack \
  bash allen_bridge/bench/build_insitu.sh
allen_bridge/bench/insitu_parkalman \
  /data/bfys/gscriven/TE_stack/PARAM/ParamFiles/data \
  allen_bridge/bench/artifacts/tracks_f32.bin \
  allen_bridge/bench/artifacts/utt_meta.bin \
  200 50 allen_bridge/bench/results/tier2_insitu.json
```

It reports `extrapUTT` and `pinn_v2_utt` µs/track (median+IQR) and their ratio.
`extrapUTT` and `pinn_v2_utt_state` are `#include`d **verbatim** from the
read-only Allen headers (`-O3 -march=native`, fp32).

## Route B — full Allen sequence monitor (GPU build, gold standard)

For a CUDA Allen build the production sequence-monitor delta is obtained as:

```bash
cd /data/bfys/gscriven/TE_stack/Allen/build.<cuda>/
source allenenv.sh
SEQ=hlt1_PbPb_PbSMOG_with_parkf            # a sequence that instantiates ParKalmanFilter
GEO=../input/detector_configuration
PARAMS=$PARAM/ParamFiles/data
MDF=../input/minbias/mdf/MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster_v1.mdf

# 1) dump the running config and expose the toggle
./Allen -g $GEO --params $PARAMS --mdf $MDF --sequence $SEQ \
        --write-configuration 1 -n 1                          # -> config.json
# 2) make two configs: kalman_filter.use_nn_utt = 0 (stock) and = 1 (NN hybrid)
python3 - <<'PY'
import json
c=json.load(open('config.json'))
for tag,val in [('stock',0),('nn',1)]:
    c2=json.loads(json.dumps(c))
    for k in c2:                       # the long-track ParKalmanFilter algorithm
        if 'kalman' in k.lower() and isinstance(c2[k],dict):
            c2[k]['use_nn_utt']=val
    json.dump(c2,open(f'config_{tag}.json','w'))
PY
# 3) run each with many repetitions; read the per-algorithm timing table
for tag in stock nn; do
  ./Allen -g $GEO --params $PARAMS --mdf $MDF \
          --sequence config_$tag.json -n 1000 -r 50 -t 1 \
          --print-status 1 | tee run_$tag.log
done
# 4) the delta in the "kalman_filter" / ParKalmanFilter algorithm line between
#    run_nn.log and run_stock.log is the in-situ +PINN overhead per event; divide
#    by tracks/event to get per-track and compare to Tier-1 PINN µs/track.
```

Notes / caveats:
- `use_nn_utt` is **not serialised** in the pre-generated sequence JSONs (it
  defaults to `false`), hence the `--write-configuration` + edit step.
- On the **CPU** build the absolute algorithm time is not the GPU production
  number, so only the **ratio / relative delta** is used as the cross-check —
  consistent with Tier-2's role as a reality check, not a second absolute number.
- The Tier-1 RK number is additionally cross-checked against Allen's published
  per-event ParKalman/HLT1 throughput (validity gate #3) in the write-up.
