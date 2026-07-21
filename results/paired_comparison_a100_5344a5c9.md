# Paired A100 campaign comparison — `5344a5c`

Canonical-runtime rerun of Role B paired activation + stability.

## Runtime

| Property | Value |
| --- | --- |
| Env | `ect-clean-validation` |
| Python / PyTorch / CUDA | 3.9.18 / 2.3.0 / 12.1 |
| GPU | NVIDIA A100-PCIE-40GB |
| Train+package HEAD | `5344a5c97ab461b640ad5c5413cbf57eec527c2a` |

## Path policy

- Runs root: `/root/ect-runs/paired-training-v1-canonical`
- `readlink -f /mnt/ect_project/runs` → `/mnt/ect_project/runs`
- `readlink -f /root/ect-runs` → `/root/ect-runs`
- Same directory: **False**
- Note: All four canonical reruns use /root/ect-runs/... exclusively. /mnt/ect_project/runs and /root/ect-runs are distinct directories (not symlinks). Prior PR #17 sigmoid activation path mismatch was caused by training under /mnt then moving the run tree to /root before packaging.

## Stability semantics

- Kind: **`independent_fresh_16_kimg`**
- Resumed from activation: **False**
- Stability arms are fresh --transfer runs with --duration=0.016, not resumes from activation training-state.

## Determinism

Configs and assets are identical across arms; CUDA/cuDNN kernels are not guaranteed bit-identical across independent processes. Pre-correction r_over_t_mean matches; early losses may differ slightly.

## Runs

| Arm | Mode | Iters / kimg | Success / skip | Final loss | Wall s | Peak MiB | Snapshot SHA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sigmoid | activation | 32 / 4.096 | 23 / 9 | 31.09924 | 37.6 | 5916.5 | `248e4638e833…` |
| adaptive_v1 | activation | 32 / 4.096 | 23 / 9 | 30.84132 | 34.1 | 5916.8 | `23a8653c34f0…` |
| sigmoid | stability | 125 / 16.0 | 116 / 9 | 26.38569 | 104.1 | 5916.3 | `32aa46615846…` |
| adaptive_v1 | stability | 125 / 16.0 | 116 / 9 | 26.70843 | 105.0 | 5916.3 | `7d162808dc98…` |

## Adaptive controller

| Mode | Signal updates | First nonzero corr | First adapted pair | Final correction | Final r/t | Final gap | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| activation | 8 | 12 | 13 | -0.0162769219282 | 0.970931 | 0.029069 | True |
| stability | 32 | 12 | 13 | -0.0247819071088 | 0.961088 | 0.038912 | None |

## Checkpoint SHAs (for Role D)

| Arm | network-snapshot-latest SHA-256 | training-state-latest SHA-256 |
| --- | --- | --- |
| sigmoid / activation | `248e4638e833389d078ec8d09726113d76d283b480479481e028b9e0560f9957` | `09b5c61d980f4d34307c7835d390ee4b740ddf97f81390565ef2678d28da2576` |
| adaptive_v1 / activation | `23a8653c34f0087561fdea250ce8501489c516119f7b7dc7f5ea5c03ae3626c4` | `ef2c93f58d7b9df944504fde66e361363312361c49fbb32ffbd0dd7fe5ad2c71` |
| sigmoid / stability | `32aa4661584663c40bc05a3b5c5bef3bd4e8b60e289023d491fbe46fac5478ed` | `0dccc15dfe5bbe49466ee18109147f1500db31f8036e120d55c70482b1e7b8f8` |
| adaptive_v1 / stability | `7d162808dc98deec28d269693cba4242b97e52d8431017fe9e45a54a8c3d151a` | `269e2ec3b0b4edadddd2bc251338995962b31ea206106b0bfa24083e7210fd83` |

Machine-readable companion: `results/paired_comparison_a100_5344a5c9.json`.
