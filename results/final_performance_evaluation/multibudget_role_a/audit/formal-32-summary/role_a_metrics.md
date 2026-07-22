# Role A quantitative evaluation

Phase: `formal`; samples per checkpoint/NFE: 5000; precision: FP32.

| Method | Train seed | Budget | NFE | KID | FID | Checkpoint SHA |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sigmoid | 0 | 32 | 1 | 0.488681257 | 435.116718791 | `cb196bf9d6ba0fdd20858cc64c59a138c625c7584a5a445a37f9bdf9380e8c71` |
| sigmoid | 0 | 32 | 2 | 0.387979180 | 353.044258189 | `cb196bf9d6ba0fdd20858cc64c59a138c625c7584a5a445a37f9bdf9380e8c71` |
| adaptive_v1 | 0 | 32 | 1 | 0.489093781 | 435.313568891 | `a9f6c06d22e1e6409680efd31d43fe90714fec4a7b211ea4bd718a6cae810d9c` |
| adaptive_v1 | 0 | 32 | 2 | 0.388893992 | 353.546742330 | `a9f6c06d22e1e6409680efd31d43fe90714fec4a7b211ea4bd718a6cae810d9c` |
| sigmoid | 1 | 32 | 1 | 0.512568593 | 441.655822030 | `e5cc717d278bcd9d5ab7930486c22ce1a81e915a502deb31e6265a3e4e126b90` |
| sigmoid | 1 | 32 | 2 | 0.215143412 | 216.320162005 | `e5cc717d278bcd9d5ab7930486c22ce1a81e915a502deb31e6265a3e4e126b90` |
| adaptive_v1 | 1 | 32 | 1 | 0.513994038 | 441.963679600 | `610f353f4afeebee9e4ab12ee1fa170365d0bb42d138aa0a06c0c087bc4e577e` |
| adaptive_v1 | 1 | 32 | 2 | 0.217176944 | 218.282160794 | `610f353f4afeebee9e4ab12ee1fa170365d0bb42d138aa0a06c0c087bc4e577e` |
| sigmoid | 2 | 32 | 1 | 0.509273708 | 448.639770619 | `3f1bf37bf6512751ab55bc89ad1971c98679c79d5b5979d20b4730d62c64e420` |
| sigmoid | 2 | 32 | 2 | 0.421430677 | 374.854800946 | `3f1bf37bf6512751ab55bc89ad1971c98679c79d5b5979d20b4730d62c64e420` |
| adaptive_v1 | 2 | 32 | 1 | 0.510767043 | 449.423978652 | `1c2e266d2e9c02459f10fbbe53c1497e8df8c656d54ea53b6c7ab69c9b3c8060` |
| adaptive_v1 | 2 | 32 | 2 | 0.428483129 | 379.053379808 | `1c2e266d2e9c02459f10fbbe53c1497e8df8c656d54ea53b6c7ab69c9b3c8060` |

Reference identity consistent: True; image count valid: True; repeat results exact: True; repeat results numerically consistent: True (rel_tol=1e-06, abs_tol=1e-12).
