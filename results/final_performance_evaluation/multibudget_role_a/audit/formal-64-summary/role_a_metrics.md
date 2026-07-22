# Role A quantitative evaluation

Phase: `formal`; samples per checkpoint/NFE: 5000; precision: FP32.

| Method | Train seed | Budget | NFE | KID | FID | Checkpoint SHA |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sigmoid | 0 | 64 | 1 | 0.487805516 | 429.590474191 | `514b67696ca911ccdd9e16c5cce93d91e2c2c77433a82a2c2b8c0ae0b410dc9f` |
| sigmoid | 0 | 64 | 2 | 0.405948251 | 364.870998714 | `514b67696ca911ccdd9e16c5cce93d91e2c2c77433a82a2c2b8c0ae0b410dc9f` |
| adaptive_v1 | 0 | 64 | 1 | 0.487248659 | 429.745804406 | `dc5fdcac27f49e845eae1d28571cd0c3be6756aa6aee785e3189c1c27f527aa3` |
| adaptive_v1 | 0 | 64 | 2 | 0.406898081 | 365.507492943 | `dc5fdcac27f49e845eae1d28571cd0c3be6756aa6aee785e3189c1c27f527aa3` |
| sigmoid | 1 | 64 | 1 | 0.475123346 | 418.230885106 | `bb98925649c0267064210a069887ae9b5dd3eff26727651318818ac2d46c09f9` |
| sigmoid | 1 | 64 | 2 | 0.284072310 | 270.979412918 | `bb98925649c0267064210a069887ae9b5dd3eff26727651318818ac2d46c09f9` |
| adaptive_v1 | 1 | 64 | 1 | 0.475448430 | 418.421415222 | `c1eaef2c89c1365aa5c3e5b65c8522152468bf213923004c299ac6a058fddbdf` |
| adaptive_v1 | 1 | 64 | 2 | 0.274910927 | 263.317393435 | `c1eaef2c89c1365aa5c3e5b65c8522152468bf213923004c299ac6a058fddbdf` |
| sigmoid | 2 | 64 | 1 | 0.519060135 | 437.319543602 | `73c5b0c7b3e5130a76f80d26cb30e6969d049d941ac0e6568ee16f7c1d5bc925` |
| sigmoid | 2 | 64 | 2 | 0.428162545 | 377.604781356 | `73c5b0c7b3e5130a76f80d26cb30e6969d049d941ac0e6568ee16f7c1d5bc925` |
| adaptive_v1 | 2 | 64 | 1 | 0.519976497 | 437.985043809 | `3f3b4a69dec396d4af1010c6c26323d442e64b60fd3c7b354c4820c60aae1157` |
| adaptive_v1 | 2 | 64 | 2 | 0.433810949 | 380.971307571 | `3f3b4a69dec396d4af1010c6c26323d442e64b60fd3c7b354c4820c60aae1157` |

Reference identity consistent: True; image count valid: True; repeat results exact: True; repeat results numerically consistent: True (rel_tol=1e-06, abs_tol=1e-12).
