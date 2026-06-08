# `loader_v3_spec.md` — V3 NN-extrapolator blob format

**Status**: locked draft, 2026-05-20 (replaces no prior version)
**Schema version**: 3 (`magic = b"NRKv3\0\0\0"`)
**Applies to**: `pinn_v2_ALLEN_v1` and any future bit-identical re-train.
**Consumers**: Python `for_allen.export.blob_writer`, Python `for_allen.export.blob_loader`, CUDA `Extrapolators::NNExtrapolator` (R6).

This document is the **single source of truth** for the byte layout. The
Python writer (`blob_writer.py`) asserts every field against this spec
before emitting bytes; the CUDA loader (`NRKExtrapolator.cuh`, R6) parses
exactly the same layout via a `constexpr` shape table. Any divergence is
a build-stopping bug.

---

## 1. Endianness, alignment, types

- **Endianness**: little-endian.
- **Alignment**: every field is naturally aligned to its own size. The header
  is padded so that the first weight block starts on a 16-byte boundary.
- **Types**:
  - `u8`  unsigned 8-bit
  - `u16` unsigned 16-bit
  - `u32` unsigned 32-bit
  - `f32` IEEE-754 binary32

`f32` is the *only* floating-point type. fp16/bf16 are **not** in scope
for R5; if they ever land they will go in a v4 blob with a new magic.

---

## 2. File layout

```
+-----------------------------+
| Section A: magic + version  |   16  bytes
+-----------------------------+
| Section B: model header     |   32  bytes
+-----------------------------+
| Section C: normalisation    |   40  bytes  (5 mean + 5 std, f32)
+-----------------------------+
| Section D: layer table      |   8 * n_layers  bytes
+-----------------------------+
| Section E: weight payload   |   variable
+-----------------------------+
| Section F: CRC32 trailer    |    4  bytes
+-----------------------------+
```

For `pinn_v2_ALLEN_v1` (the only current consumer):
`n_layers = 3`, total file size = **41 604 bytes** (40.63 kiB) — under the
64 kiB cap.

Byte breakdown: 16 (§A) + 32 (§B) + 40 (§C) + 24 (§D, no pad needed) +
41 488 (§E weights) + 4 (§F CRC32) = **41 604 B**.

---

### 2.A Magic + version (16 B)

| offset | size | type | name        | value                |
|-------:|-----:|------|-------------|----------------------|
|      0 |    8 | u8[8]| `magic`     | `b"NRKv3\0\0\0"`     |
|      8 |    4 | u32  | `version`   | `3`                  |
|     12 |    4 | u32  | `_pad0`     | `0` (must be zero)   |

### 2.B Model header (32 B)

| offset | size | type | name              | meaning                                 |
|-------:|-----:|------|-------------------|-----------------------------------------|
|     16 |    4 | u32  | `arch_id`         | `1 = PINN_V2_DIPOLE` (only valid value) |
|     20 |    4 | u32  | `activation_id`   | `1 = TANH` (only valid value)           |
|     24 |    4 | u32  | `input_dim`       | `7` (physical input layout)             |
|     28 |    4 | u32  | `output_dim`      | `5` (physical output layout)            |
|     32 |    4 | u32  | `encoder_in_dim`  | `6` (5 normalised state + 1 z_frac)     |
|     36 |    4 | u32  | `n_norm`          | `5` (only first 5 inputs are normalised)|
|     40 |    4 | u32  | `n_layers`        | number of `nn.Linear` blocks (= 3)      |
|     44 |    4 | u32  | `total_params`    | sanity counter, must match payload      |

**Enum tables** (frozen, never re-numbered):

```
arch_id        : 0 = INVALID,  1 = PINN_V2_DIPOLE
activation_id  : 0 = INVALID,  1 = TANH,  2 = SILU (reserved),  3 = RELU (reserved)
```

### 2.C Normalisation (40 B)

The network only z-score-normalises the **first 5** input channels
(`x, y, tx, ty, qop`). The remaining channels (`z_start`, `dz`) are
consumed outside the network — `z_start` is unused at inference;
`dz` flows through the envelope (§3) directly.

| offset | size | type    | name         |
|-------:|-----:|---------|--------------|
|     48 |   20 | f32[5]  | `input_mean` |
|     68 |   20 | f32[5]  | `input_std`  |

`input_std[i]` must be strictly positive for all `i`. The writer asserts
this; the loader must assert it too.

### 2.D Layer table (8 × n_layers B)

Repeats `n_layers` times, starting at byte 88:

| field | size | type | meaning                                             |
|-------|-----:|------|-----------------------------------------------------|
| `out` |    4 | u32  | output dimension of this `nn.Linear`                |
| `in`  |    4 | u32  | input  dimension of this `nn.Linear`                |

For `pinn_v2_ALLEN_v1` the table is exactly:

```
layer 0 :  out=96, in=6
layer 1 :  out=96, in=96
layer 2 :  out= 4, in=96
```

Total table size = 24 B → §2.E starts at byte **112**. The padding
rule is uniform: after the layer table, the writer emits
`pad_bytes = (-raw_payload_offset) % 16` zero bytes, where
`raw_payload_offset = 88 + 8 * n_layers`. For `n_layers = 3`,
`raw_payload_offset = 112` is already 16-B-aligned, so `pad_bytes = 0`.
For any odd `n_layers`, `pad_bytes = 8`. **The writer enforces this
rule explicitly; the CUDA loader must compute the same expression at
load time — it must NOT hardcode `pad_bytes = 0`.**

### 2.E Weight payload

For each layer `ℓ`, in declaration order:

1. `W[ℓ]` : `f32[out × in]` — **row-major** (`W[i, j]` at offset
   `i * in + j`). This matches PyTorch's `nn.Linear.weight` storage:
   in PyTorch `y = x @ W.T + b`, so `W.shape = (out, in)` and the
   row-major flatten of `W` is what we write.
2. `b[ℓ]` : `f32[out]`.

Concretely for `pinn_v2_ALLEN_v1`:

```
L0 weight :  96 * 6 = 576    f32   ( 2304 B)
L0 bias   :  96             f32   (  384 B)
L1 weight :  96 * 96 = 9216  f32   (36864 B)
L1 bias   :  96             f32   (  384 B)
L2 weight :   4 * 96 = 384   f32   ( 1536 B)
L2 bias   :   4             f32   (   16 B)
-------------------------------------------
total payload                       41488 B
total params                        10372
```

### 2.F CRC32 trailer

The last 4 bytes of the file are a **CRC32** (polynomial 0x04C11DB7,
init 0xFFFFFFFF, reflected — i.e. Python's `zlib.crc32`) over **all
preceding bytes** (sections A–E). The loader recomputes and rejects
the blob on mismatch.

---

## 3. Inference contract (the CUDA loader's obligation)

Given input `x[7] = [x, y, tx, ty, qop, z_start, dz]` (physical units,
mm / dimensionless / c·MeV⁻¹ / mm / mm), output `y[5] =
[x_f, y_f, tx_f, ty_f, qop_f]`:

```
# normalise
for i in 0..4 :  x_norm[i] = (x[i] - input_mean[i]) / input_std[i]

# encoder input = [x_norm[0..4],  z_frac = 1.0]
h0 = concat(x_norm[0..4], 1.0)                         # 6 dims

# layer 0: h1 = tanh( W0 @ h0 + b0 )                    # 96 dims
# layer 1: h2 = tanh( W1 @ h1 + b1 )                    # 96 dims
# layer 2:  c = W2 @ h2 + b2                            #  4 dims (NO activation)

# envelope (NOT learned, constexpr in CUDA)
y[0] = x[0] + x[2]*dz + c[2]*dz                        #   x_f
y[1] = x[1] + x[3]*dz + c[3]*dz                        #   y_f
y[2] = x[2]           + c[0]                           #  tx_f
y[3] = x[3]           + c[1]                           #  ty_f
y[4] = x[4]                                            # qop_f (passthrough)
```

`z_frac` is **always 1.0** at inference; it is only varied internally
by training-time PINN collocation. The CUDA loader hardcodes it.

All four operations in the envelope are FMA-able; the CUDA
implementation **must** use `fmaf` to match the Python reference
to within 1 ULP (see §4).

---

## 4. Bit-bound parity gate (the R5 verification gate)

The R5 phase is considered complete when, on the 200-track A4
reference set (`artifacts/phase1a/X_a4.npy`):

```
max | y_python_fp32  −  y_cuda_fp32 |   <   1 ULP
```

evaluated independently for each of the 5 output components. Python
reference is `for_allen.export.blob_loader` reading the same blob the
CUDA loader reads — i.e. neither path is allowed to consult the
original `best_model.pt` or `normalization.json`. This is what makes
the blob the canonical artefact.

> **Reviewer note.** "Within 1 ULP" is the strongest gate that does
> not require enforcing a global FP evaluation order on the CUDA side.
> If a future kernel uses a different reduction order or fuses
> matrix-vector products into a single `mma` instruction, this gate
> relaxes to "within 4 ULP and median 0 ULP". Any further relaxation
> requires an ADR.

---

## 5. Forbidden by this version

- fp16 / bf16 weights — reserved for a v4 blob.
- Multiple sub-networks in one blob — reserved for a future "multi-arch"
  schema (e.g. one blob carrying both `PINN_V2_DIPOLE` and
  `PINN_V2_UTT`).
- Compression — the 64 kiB budget makes this pointless and would
  defeat the CRC32.
- Variable activation per layer — a single `activation_id` applies to
  *all* hidden layers; the output layer is always linear.
- Endian swap — little-endian only. A big-endian Allen host would
  trip the magic check.

---

## 6. Change control

Any modification to this file requires:
1. A bump of `magic[3]` (i.e. `NRKv3` → `NRKv4`) **and** the `version` u32.
2. A new ADR under `docs/decisions/`.
3. The Python writer rejecting the old `version` value at write time.
4. A new bit-bound parity test for the new version; the old test is
   retained as a regression guard.

The spec is **append-only** within a major version: fields may be added
*after* §2.E (before the CRC32) by extending `n_layers` semantics or
introducing a new section between E and F, but only behind a
back-compatible flag word. In practice, just bump the version.
