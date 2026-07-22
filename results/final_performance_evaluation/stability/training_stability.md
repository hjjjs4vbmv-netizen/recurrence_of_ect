# Training stability summary

| Schedule | Seed | kimg | Attempts | Successful | Skipped | Finite loss | Final scale | Peak VRAM MiB | Wall time s | Adaptive active |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| sigmoid | 0 | 16.000 | 125 | 116 | 9 | yes | 128 | 5916.3 | 104.1 | n/a |
| adaptive_v1 | 0 | 16.000 | 125 | 116 | 9 | yes | 128 | 5916.3 | 105.0 | true |
| sigmoid | 1 | 16.000 | 125 | 116 | 9 | yes | 128 | 5914.4 | 105.3 | n/a |
| adaptive_v1 | 1 | 16.000 | 125 | 116 | 9 | yes | 128 | 5916.3 | 102.2 | true |
| sigmoid | 2 | 16.000 | 125 | 116 | 9 | yes | 128 | 5916.3 | 101.9 | n/a |
| adaptive_v1 | 2 | 16.000 | 125 | 116 | 9 | yes | 128 | 5914.4 | 102.0 | true |

All six 16 kimg runs complete: **True**. All recorded losses finite: **True**.

Skipped AMP steps, GradScaler values, time, and memory are engineering stability descriptors; they are not generation-quality metrics.
