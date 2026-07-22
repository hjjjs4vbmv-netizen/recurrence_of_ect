# Role A quantitative evaluation

Phase: `formal`; samples per checkpoint/NFE: 5000; precision: FP32.

| Method | Train seed | Budget | NFE | KID | FID | Checkpoint SHA |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sigmoid | 0 | 16 | 1 | 0.591114998 | 483.669642034 | `32aa4661584663c40bc05a3b5c5bef3bd4e8b60e289023d491fbe46fac5478ed` |
| sigmoid | 0 | 16 | 2 | 0.243654370 | 238.356426634 | `32aa4661584663c40bc05a3b5c5bef3bd4e8b60e289023d491fbe46fac5478ed` |
| adaptive_v1 | 0 | 16 | 1 | 0.594804227 | 484.782869039 | `7d162808dc98deec28d269693cba4242b97e52d8431017fe9e45a54a8c3d151a` |
| adaptive_v1 | 0 | 16 | 2 | 0.245124236 | 239.596419159 | `7d162808dc98deec28d269693cba4242b97e52d8431017fe9e45a54a8c3d151a` |
| sigmoid | 1 | 16 | 1 | 0.437476933 | 381.841896410 | `3ffd271c99a3d09bc63c749f5b69f76af37bbbec54678e88885eb3341750e7ec` |
| sigmoid | 1 | 16 | 2 | 0.284793675 | 278.535449689 | `3ffd271c99a3d09bc63c749f5b69f76af37bbbec54678e88885eb3341750e7ec` |
| adaptive_v1 | 1 | 16 | 1 | 0.427847743 | 375.531957609 | `b7bc291a80298121c091945519d79f1f8c79deb318b4790e2d7371a1510b7a13` |
| adaptive_v1 | 1 | 16 | 2 | 0.285328060 | 278.940941940 | `b7bc291a80298121c091945519d79f1f8c79deb318b4790e2d7371a1510b7a13` |
| sigmoid | 2 | 16 | 1 | 0.567756176 | 475.878016256 | `3fed09e540f8419db32bac9c1fdfe93033aaad225198ecc2bbce4eccf26b5865` |
| sigmoid | 2 | 16 | 2 | 0.358875096 | 330.291874258 | `3fed09e540f8419db32bac9c1fdfe93033aaad225198ecc2bbce4eccf26b5865` |
| adaptive_v1 | 2 | 16 | 1 | 0.573519409 | 478.220183356 | `b04098a8d53449b3d2666f87b8d63e6f16ba11d6c2a5069c1d81d33d995f1d43` |
| adaptive_v1 | 2 | 16 | 2 | 0.360325783 | 331.161000533 | `b04098a8d53449b3d2666f87b8d63e6f16ba11d6c2a5069c1d81d33d995f1d43` |

Reference identity consistent: True; image count valid: True; repeat results exact: True; repeat results numerically consistent: True (rel_tol=1e-06, abs_tol=1e-12).
