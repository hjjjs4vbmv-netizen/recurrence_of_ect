# Evaluation protocol

This document defines the reproducible Role D sampling protocol. It separates
historical evidence from results produced under the current protocol.

## Result classes

`results/preliminary_seed42_fp32_8ksteps/` contains preliminary historical
results. They were produced from an older code base and an approximately
8k-update checkpoint. They are retained for reference only, are not directly
comparable with the current B/C protocol, and must not be reported as a final
benchmark.

`results/fixed_seeds_0_63_fp32_8ksteps/` is also a preliminary historical
smoke from that checkpoint. Its directory-level README records why it does not
satisfy the current protocol. It must not be regenerated or treated as a
current-protocol result.

Current protocol results must record the checkpoint SHA256, evaluation Git
commit, seeds, NFE, `mid_t`, precision, GPU, work-group sizes, and repeated-run
determinism status in `metadata.json`. They must also record elapsed time, image
and seed counts, the complete seed list, per-mode NFE and `mid_t`, generator
implementation, and the actual forward batch size.

## Fixed-seed protocol

- Seeds 0-63 denote 64 per-sample seeds, not 64 repeated metric runs.
- Generate one 32x32 RGB PNG for each seed in 0-63.
- NFE=1 uses `mid_t=[]`.
- NFE=2 uses `mid_t=[0.821]`.
- For a given seed, NFE=1 and NFE=2 use the same initial latent.
- The NFE=2 intermediate noise is also deterministically derived from that seed.
- The 64 images per mode are used for visualization and determinism checks.
- Keep the model forward batch size at one.
- Treat 8 and 16 as work-group sizes, not model batch sizes.
- Require pixel-identical output across work-group sizes 8 and 16.
- Repeat each NFE configuration and require pixel-identical output.
- Isolate every result directory using the checkpoint filename and the first 12
  characters of its SHA256.
- Select precision explicitly. Use `fp32` for this acceptance smoke; use
  `checkpoint` only when intentionally preserving checkpoint-native precision.

The sampler writes to `<outdir>/<checkpoint-stem>-<sha256-prefix>/`. For
example, the official checkpoint is written under
`edm-cifar10-32x32-uncond-vp-4d5dcc1f1d0d/`. A run fails before publishing
metadata if either work-group or repeated-run determinism fails.

Each checkpoint directory contains:

```text
nfe1/
  images/
  grid_8x8.png
nfe2/
  images/
  grid_8x8.png
metadata.json
sha256_manifest.txt
```

The metadata schema records the evaluation Git commit, checkpoint path and
SHA256, checkpoint ID, seed list, NFE modes, `mid_t` per mode, precision,
device, GPU, elapsed time, image counts, generator implementation, actual
model forward batch size, verified work-group sizes, image dimensions, and the
overall determinism result.

## Official EDM checkpoint smoke

Download and verify the official NVIDIA EDM CIFAR-10 32x32 unconditional VP
checkpoint:

```bash
bash download_checkpoint.sh \
  --output /mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl
```

The expected checkpoint SHA256 is:

```text
4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da
```

Run the 64-image NFE=1 plus 64-image NFE=2 smoke:

```bash
bash scripts/sample_checkpoint.sh \
  /mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
  --outdir /mnt/ect_project/evaluations \
  --seeds 0-63 \
  --nfe 1 2 \
  --mid-t 0.821 \
  --work-group-size 8 \
  --verify-work-group-size 16 \
  --precision fp32 \
  --device cuda
```

The expected output root is:

```text
/mnt/ect_project/evaluations/edm-cifar10-32x32-uncond-vp-4d5dcc1f1d0d/
```

Verify the image counts and manifest after the command completes:

```bash
RESULT=/mnt/ect_project/evaluations/edm-cifar10-32x32-uncond-vp-4d5dcc1f1d0d
find "$RESULT/nfe1/images" -type f -name '*.png' | wc -l
find "$RESULT/nfe2/images" -type f -name '*.png' | wc -l
(cd "$RESULT" && sha256sum -c sha256_manifest.txt)
```

Both image counts must be 64 and every manifest entry must report `OK`.

## Metric boundary

This acceptance smoke does not run FID-50k, KID-50k, or any other distribution
metric. In particular, generating seeds 0-63 does not mean running FID 64
times. Its purpose is limited to checkpoint loading, fixed-seed generation,
visualization, output completeness, work-group invariance, and repeated-run
determinism. Historical seed42 FP32 FID/KID values remain preliminary evidence
and are not directly comparable with current B/C formal results. Formal
FID-50k runs only after the final checkpoint and comparison protocol are
frozen.

## Git artifact policy

Do not commit the checkpoint or the 128 individual PNG files. For a completed
smoke run, commit only both 8x8 grids, `metadata.json`, and
`sha256_manifest.txt`. Keep the full output tree under
`/mnt/ect_project/evaluations/` as the external evaluation record.
