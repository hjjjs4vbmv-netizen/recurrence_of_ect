# Role A quantitative evaluation protocol

This protocol supersedes the earlier 16 kimg-only evaluation scope. Role A owns
metric execution only; training jobs may continue independently.

## Frozen matrix and order

Formal evaluation is performed independently for every checkpoint and NFE. The
execution order is:

1. all six 64 kimg checkpoints;
2. all six 32 kimg checkpoints;
3. all six 16 kimg checkpoints.

Each budget must contain fixed sigmoid and Adaptive v1 for training seeds 0, 1,
and 2. The runner refuses partial six-cell budget matrices. Generated features
from different training seeds are never pooled into one metric.

## Metric smoke

Before formal evaluation, use the two existing seed0/16 kimg checkpoints:

- NFE=1 and NFE=2 (`mid_t=0.821`);
- FP32;
- 512 generated samples with explicit sampling seeds 0-511;
- `kid512_full` and `fid512_full`;
- two exact repeats per checkpoint/NFE.

The smoke collector fails unless all four checkpoint/NFE cells finish, every
metric has exactly two results, the declared image count equals the sampling
seed count, and repeat values match exactly. The run manifest records method,
training seed, budget, NFE, checkpoint SHA256, dataset SHA256, sampling seeds,
precision, metric seed, and reference feature extractor.

Smoke KID/FID values are diagnostic only and must not be reported as formal
generation-quality results.

## Formal evaluation

For each checkpoint and each NFE:

- 5,000 generated samples;
- identical explicit sampling seeds 0-4999;
- FP32;
- NFE=1 or NFE=2 (`mid_t=0.821`);
- KID-5k and FID-5k when both are stable.

The required output columns are exactly:

| Method | Train seed | Budget | NFE | KID | FID | Checkpoint SHA |
| --- | ---: | ---: | ---: | ---: | ---: | --- |

All 5k metrics are proxy evaluations, not standard FID-50k benchmarks.

## Uniform fallback rule

Metric fallback is selected for an entire complete budget matrix, never per
method or checkpoint:

1. `--metrics both`: report KID and FID;
2. if FID is unstable but KID is stable, start a new empty run with
   `--metrics kid-only`;
3. if KID is unstable but FID is stable, start a new empty run with
   `--metrics fid-only` and label it FID-5k proxy;
4. never combine a fixed result from one metric mode with an Adaptive result
   from another metric mode.

Failed runs are not retried automatically.

## Reference identity on the current A100 node

- CIFAR-10 ZIP SHA256:
  `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`
- Inception detector SHA256:
  `f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4`
- Cached reference statistics SHA256:
  `7c7ad1657d62a12ac6bb609ccd2da79dcefde1aff48d2883bba1c6556b4671b3`
  and
  `c9e49db82db2c299bc01b415f55beed0deb53782629457f97905ecbd66b60870`

All jobs must use this same dataset, detector, and reference-statistics
identity.

## Commands

Smoke:

```bash
python scripts/run_role_a_quality_evaluation.py \
  --manifest /root/role-a-eval/checkpoints.json \
  --data /mnt/ect_project/datasets/cifar10-32x32.zip \
  --outdir /root/role-a-eval/smoke-both \
  --phase smoke --budget 16 --metrics both

python scripts/collect_role_a_quality_results.py \
  --eval-root /root/role-a-eval/smoke-both \
  --outdir /root/role-a-eval/smoke-summary
```

Formal evaluation is invoked separately in the frozen order by replacing
`BUDGET` with 64, then 32, then 16:

```bash
python scripts/run_role_a_quality_evaluation.py \
  --manifest /root/role-a-eval/checkpoints.json \
  --data /mnt/ect_project/datasets/cifar10-32x32.zip \
  --outdir /root/role-a-eval/formal-BUDGET-both \
  --phase formal --budget BUDGET --metrics both
```

Each output directory must be new and empty.
