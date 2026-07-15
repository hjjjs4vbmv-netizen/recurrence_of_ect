# Preliminary seed42 FP32 results

These files are **preliminary historical results** preserved from Role D's first-day evaluation. They were produced on old base commit `4311059770f54821d151a9b0e1f76770a5f3930e` and originally recorded in commit `52b1f2350744af4f6fbee3142b21fea2bc62f0b1`.

Protocol summary:

- CIFAR-10 32x32, unconditional generation
- FP32 training and evaluation
- training seed 42 and evaluation seed 42
- approximately 8k training updates, inferred from the run directory name
- NFE=1 and NFE=2, with `mid_t=0.821` for NFE=2
- NVIDIA A100-PCIE-40GB

The checkpoint is stored outside Git. Its historical path is recorded in `checkpoint.txt`, but its SHA256 was not captured with the original result and remains unknown.

These measurements are not directly comparable to the current B/C protocol and must not be treated as a final benchmark. They are retained only as preliminary evidence and as a reference for the reproducible evaluation implementation.
