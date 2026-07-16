# Preliminary historical fixed-seed smoke

This directory contains **preliminary historical results** from the Role D
8k-step FP32 checkpoint. It predates `docs/EVALUATION_PROTOCOL.md` and is kept
only as evidence that the initial fixed-seed workflow ran successfully.

These artifacts are not a formal benchmark and are not directly comparable to
future B/C results. In particular, they were generated before repeated-run
determinism, checkpoint-SHA output isolation, and the final metadata schema were
required. The checkpoint remains outside Git.

Do not regenerate or reinterpret this directory as a current-protocol result.
Current smoke outputs must use the checkpoint-isolated layout defined in
`docs/EVALUATION_PROTOCOL.md`.
