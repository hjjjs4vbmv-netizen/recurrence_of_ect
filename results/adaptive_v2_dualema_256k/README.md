# Adaptive v2 Dual-EMA — paired 256 kimg evidence

This directory contains the small, auditable artifacts from the frozen
`adaptive_v2_dualema_paired_256k_v1` experiment. Large network snapshots and
optimizer states remain on the training host; their SHA-256 digests are stored
in `training-validations/`.

## Frozen identity

- Git commit: `dfeb04db3907888d15883f6c8ac079e5207e8aa0`
- Branch: `role-c/adaptive-v2-dualema-256k`
- Device: NVIDIA GeForce RTX 5090, CUDA 12.8, PyTorch 2.10.0+cu128
- Dataset SHA-256: `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`
- Transfer checkpoint SHA-256: `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`
- Matrix: sigmoid and adaptive v2, training seeds 0/1/2, 256 kimg each
- Adaptive defaults: beta-fast 0.80, beta-slow 0.98, max-adjust 0.05,
  warmup 8 successful signal updates, epsilon 1e-8

The exact environment and protocol are recorded in `FROZEN_PROTOCOL.txt`; the
12-cell evaluation commands and checkpoint hashes are in
`evaluation_manifest.json`.

## Validation

- Full unit suite: 137/137 tests passed.
- Fresh 4 kimg plus numbered-state resume smoke: PASS. The first nonzero
  correction and first adapted pair both occurred at iteration 33; no
  non-finite controller signal was observed.
- Formal training: all six runs PASS. Each run attempted 2,000 iterations,
  completed 1,992 optimizer steps, and skipped the same eight initial AMP
  scaler steps. All 64/128/256 kimg network and state hashes are recorded.
- During training, throughput was approximately 1.68 sec/kimg. A live sample
  measured 98% GPU utilization, about 9.6 GiB peak training memory, and about
  529--532 W. Evaluation sampled 96% utilization at about 472 W.

## Controller behavior

All three adaptive runs began changing pairs at iteration 33. Corrections were
bidirectional, with 71/74/81 sign flips for seeds 0/1/2. Correction saturation
ratios were 0.8%/0.0%/0.2%, and no NaN or Inf signal was recorded. The full
per-seed telemetry and plots are in `telemetry-summary/`.

## 256 kimg quality result

Both metrics use 5,000 generated samples with fixed sample seeds 0--4999 in
FP32. NFE=2 uses `mid_t=0.821`. Values are mean +/- population standard
deviation over the three paired training seeds; lower is better. FID-5k is a
proxy and must not be described as FID-50k.

| Metric | NFE | sigmoid | adaptive v2 | adaptive - sigmoid | Seeds improved |
|---|---:|---:|---:|---:|---:|
| KID-5k | 1 | 0.216528 +/- 0.002449 | 0.224685 +/- 0.006154 | +0.008158 (+3.77%) | 0/3 |
| KID-5k | 2 | 0.037991 +/- 0.000418 | 0.039160 +/- 0.000413 | +0.001169 (+3.08%) | 0/3 |
| FID-5k proxy | 1 | 222.924 +/- 1.845 | 229.010 +/- 4.967 | +6.086 (+2.73%) | 0/3 |
| FID-5k proxy | 2 | 51.604 +/- 0.426 | 52.560 +/- 0.310 | +0.956 (+1.85%) | 0/3 |

Adaptive v2 was worse in all 12 paired metric comparisons. The pre-registered
quality criterion therefore failed: **no quality advantage is established**.
The experiment does support the narrower conclusion that the controller ran
stably and resumed correctly under the frozen protocol.

## Artifact map

- `SMOKE_VALIDATION.json`: fresh/resume smoke assertions.
- `training-validations/`: six formal-run status files and checkpoint hashes.
- `telemetry-summary/`: controller CSV/JSON and three trajectory plots.
- `evaluation_manifest.json`: exact 12-cell evaluation identity and commands.
- `quality-summary/`: raw per-seed values, paired differences, aggregates, and
  the machine-readable pre-registered conclusion.
