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

- Generate one 32x32 RGB PNG for each seed in 0-63.
- Run NFE=1 and NFE=2, using `mid_t=0.821` for NFE=2.
- Keep the model forward batch size at one.
- Treat 8 and 16 as work-group sizes, not model batch sizes.
- Require pixel-identical output across work-group sizes 8 and 16.
- Repeat each NFE configuration and require pixel-identical output.
- Isolate every result directory by the full checkpoint SHA256.
- Select precision explicitly. Use `fp32` for this acceptance smoke; use
  `checkpoint` only when intentionally preserving checkpoint-native precision.

The sampler writes to `<outdir>/<checkpoint_sha256>/`. A run fails before
publishing metadata if either work-group or repeated-run determinism fails.

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
/mnt/ect_project/evaluations/4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da/
```

Verify the image counts and manifest after the command completes:

```bash
RESULT=/mnt/ect_project/evaluations/4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da
find "$RESULT/nfe1/images" -type f -name '*.png' | wc -l
find "$RESULT/nfe2/images" -type f -name '*.png' | wc -l
(cd "$RESULT" && sha256sum -c sha256_manifest.txt)
```

Both image counts must be 64 and every manifest entry must report `OK`.

## Metric boundary

This acceptance smoke does not run FID-50k, KID-50k, or any other distribution
metric. Its purpose is limited to checkpoint loading, fixed-seed generation,
output completeness, work-group invariance, and repeated-run determinism.
Historical FID/KID values remain preliminary evidence only. Formal metrics are
deferred until the B/C checkpoint and comparison protocol are finalized.

## Git artifact policy

Do not commit the checkpoint or the 128 individual PNG files. For a completed
smoke run, commit only both 8x8 grids, `metadata.json`, and
`sha256_manifest.txt`. Keep the full output tree under
`/mnt/ect_project/evaluations/` as the external evaluation record.
