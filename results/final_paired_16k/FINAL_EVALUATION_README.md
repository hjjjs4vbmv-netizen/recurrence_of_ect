# Final paired 16 kimg fixed-seed evaluation

## Material Passport

- Mode: `validate`
- Verification status: `VERIFIED`
- Training commit: `5344a5c97ab461b640ad5c5413cbf57eec527c2a`
- Evidence/evaluation commit: `ef4aa3142eb2049bd5d541d18d7278cdf758029c`
- Evaluation environment: `ect-clean-validation`
- Runtime: Python 3.9.18, PyTorch 2.3.0, CUDA 12.1
- GPU: NVIDIA A100-PCIE-40GB
- Data access: compact grids and metadata are committed; individual PNGs and full logs remain on persistent storage

## Checkpoints

| Method | Persistent checkpoint | SHA-256 |
| --- | --- | --- |
| Sigmoid fixed schedule | `/mnt/ect_project/checkpoints/paired_16k_5344a5c_canonical/sigmoid_16k_canonical.pkl` | `32aa4661584663c40bc05a3b5c5bef3bd4e8b60e289023d491fbe46fac5478ed` |
| Adaptive v1 schedule | `/mnt/ect_project/checkpoints/paired_16k_5344a5c_canonical/adaptive_v1_16k_canonical.pkl` | `7d162808dc98deec28d269693cba4242b97e52d8431017fe9e45a54a8c3d151a` |

Both checkpoint hashes were verified immediately before sampling.

## Evaluation protocol

- Seeds: `0-63`, exactly once per mode
- NFE modes: `1` and `2`
- NFE=2 intermediate time: `mid_t=0.821`
- Precision: FP32
- Device: CUDA
- Model forward batch size: 1
- Primary work-group size: 8
- Verification work-group size: 16
- Repeat runs verified by the sampler: 2
- Image format: RGB, 32×32 PNG
- FID-50k/KID-50k: not run

The sampler generated every image once for work-group size 8, regenerated it
for work-group size 16, and generated it again for the repeat-run check. It
raises an error before writing final evidence if any corresponding pixels
differ.

## Verified results

| Method | NFE=1 images | NFE=2 images | Manifest entries | Total elapsed | Determinism | Work groups |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Sigmoid fixed schedule | 64 | 64 | 131 | 11.157285 s | PASS | 8/16 pixel-identical |
| Adaptive v1 schedule | 64 | 64 | 131 | 11.209927 s | PASS | 8/16 pixel-identical |

Independent post-run validation confirmed:

- both seed lists are exactly `0-63`, with no missing or duplicate seeds;
- all 256 individual outputs are RGB 32×32 PNGs;
- all four method/NFE groups contain exactly 64 images;
- both 8×8 grids are RGB 256×256 images;
- each result manifest contains exactly 128 individual images, two grids, and
  one metadata file;
- every manifest entry matches the corresponding persistent file;
- both metadata files record the required evaluation commit, checkpoint hash,
  NFE modes, `mid_t`, FP32 precision, A100 device, image counts, repeat-run
  verification, and work-group verification.

## Visual evidence

Each method directory contains its NFE=1 and NFE=2 8×8 grid. The two comparison
grids arrange eight seed pairs per row and eight rows in total. Within every
pair, the Sigmoid/fixed output is on the left and the Adaptive v1 output is on
the right; seeds are ordered `0-63` row-major.

| Comparison | SHA-256 |
| --- | --- |
| `comparison_nfe1_fixed_vs_adaptive.png` | `d95f4bdea4bb25f6cc735f60ed1df1182bb34f809a3e701ebcbc09fee8a10cc6` |
| `comparison_nfe2_fixed_vs_adaptive.png` | `52307e52e24505f09f47760142a140803b17aec81de3cd4354a43dc2180f092b` |

For the same seed, the fixed and adaptive PNG hashes differ for all 64 seeds at
NFE=1 and all 64 seeds at NFE=2. This establishes that the two checkpoints do
not produce byte-identical outputs under the shared protocol. It does **not**
establish that either method has better generation quality.

## Persistent evidence and manifest scope

The complete evaluation output remains at:

```text
/mnt/ect_project/evaluations/final_paired_16k/
├── sigmoid_16k_canonical-32aa46615846/
├── adaptive_v1_16k_canonical-7d162808dc98/
├── comparison_nfe1_fixed_vs_adaptive.png
└── comparison_nfe2_fixed_vs_adaptive.png
```

The committed `sha256_manifest.txt` files describe the complete persistent
result directories, including the individual PNGs that are intentionally not
committed. Validate them on the evaluation node with:

```bash
cd /mnt/ect_project/evaluations/final_paired_16k/sigmoid_16k_canonical-32aa46615846
sha256sum -c sha256_manifest.txt

cd /mnt/ect_project/evaluations/final_paired_16k/adaptive_v1_16k_canonical-7d162808dc98
sha256sum -c sha256_manifest.txt
```

## Interpretation boundary

This evaluation verifies checkpoint identity, deterministic fixed-seed
sampling, work-group independence, output completeness, and side-by-side visual
availability. No FID, KID, statistical quality comparison, or superiority claim
is included. Statistical fallacy scanning is not applicable because no
inferential statistics or quality metrics were computed.
