# Role A quantitative evaluation

Phase: `smoke`; samples per checkpoint/NFE: 512; precision: FP32.

| Method | Train seed | Budget | NFE | KID | FID | Checkpoint SHA |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sigmoid | 0 | 16 | 1 | 0.592271864 | 492.778948659 | `32aa4661584663c40bc05a3b5c5bef3bd4e8b60e289023d491fbe46fac5478ed` |
| sigmoid | 0 | 16 | 2 | 0.243605286 | 266.866660258 | `32aa4661584663c40bc05a3b5c5bef3bd4e8b60e289023d491fbe46fac5478ed` |
| adaptive_v1 | 0 | 16 | 1 | 0.595818043 | 494.234855438 | `7d162808dc98deec28d269693cba4242b97e52d8431017fe9e45a54a8c3d151a` |
| adaptive_v1 | 0 | 16 | 2 | 0.245059222 | 267.788344869 | `7d162808dc98deec28d269693cba4242b97e52d8431017fe9e45a54a8c3d151a` |

Reference identity consistent: True; image count valid: True; repeat results exact: False; repeat results numerically consistent: True (rel_tol=1e-06, abs_tol=1e-12).
