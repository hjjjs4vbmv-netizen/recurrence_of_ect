# Day 1 A：工程环境、资产与连通性验证

本文档对应工程和环境复现工作线。当前整合以公共基线
`origin/leader/day1-bootstrap@4e33194777a347ea5286b5ec1d5c29a58c792d29`
为基础，并保留该基线的 AMP、GradScaler 和 `metrics=none` 行为。

这里的 100-step smoke 只验证工程连通性，不是官方固定 ECT baseline，
也不能用于报告正式训练质量、FID 或 KID。

## 1. 持久化目录

脚本默认使用以下布局：

```text
/mnt/ect_project/
├── datasets/
├── pretrained/
├── runs/
└── checkpoints/
```

默认资产路径为：

```text
/mnt/ect_project/datasets/cifar-10-python.tar.gz
/mnt/ect_project/datasets/cifar10-32x32.zip
/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl
```

如平台的持久盘挂载点不同，可统一设置：

```bash
export ECT_PROJECT_ROOT=/path/to/persistent/ect_project
```

也可以分别覆盖：

```bash
export ECT_CIFAR10_TARBALL=/path/to/cifar-10-python.tar.gz
export ECT_DATA_PATH=/path/to/cifar10-32x32.zip
export ECT_TRANSFER_PATH=/path/to/edm-cifar10-32x32-uncond-vp.pkl
export ECT_RUNS_ROOT=/path/to/persistent/runs
```

## 2. 环境检查

`env.yml` 仍由公共基线管理，本次工程整合不会覆盖它。MatrixCloud/矩池云可用
`conda-matpool.yml` 处理 channel 映射。

已有 `ect` 环境时只检查，不更新：

```bash
bash setup_env.sh --check-only
```

首次创建环境：

```bash
bash setup_env.sh
```

只有明确决定同步依赖时才使用：

```bash
bash setup_env.sh --update
```

检查器记录 Python、包版本、import、CUDA、GPU、Git SHA 和工作树状态。公共基线已
验证的关键版本包括 Python 3.9.18、PyTorch 2.3.0、CUDA 12.1、diffusers
0.26.3、accelerate 0.27.2 和 huggingface-hub 0.23.4。

## 3. CIFAR-10 准备与验证

只检查现有持久化资产：

```bash
bash prepare_data.sh --check-only
```

资产不存在时才执行下载和转换：

```bash
bash prepare_data.sh
```

数据验收包括：

1. 官方原始 tarball MD5 `c58f30108f718f92721af3b95e74349a`；
2. ZIP CRC；
3. 恰好 50000 个 PNG；
4. 恰好 50000 个 labels，且与 PNG 一一对应；
5. 所有 PNG 均为 32×32 RGB；
6. `ImageFolderDataset` 长度为 50000，且首、中、尾样本可读取；
7. 记录转换后 ZIP SHA256，但不把单一 SHA256 作为内容一致性的硬约束。

转换 ZIP 可能因 ZIP entry 时间戳不同而具有不同 SHA256；因此原始 tarball MD5、
CRC、图像/标签内容和项目数据加载器检查才是主要依据。

## 4. 官方 EDM transfer checkpoint

只检查现有 checkpoint：

```bash
bash download_checkpoint.sh --check-only
```

文件不存在时才下载：

```bash
bash download_checkpoint.sh
```

默认目标位于 `/mnt/ect_project/pretrained/`，并检查官方 SHA256：

```text
4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da
```

## 5. 工程 smoke 与正式 baseline 的边界

新脚本名称为：

```text
scripts/smoke_engineering_100steps.sh
```

它的职责仅包括：

- 环境和资产可读取；
- `ct_train.py` 配置连通；
- 经单独授权后可执行 fresh 100 updates；
- 经单独授权后可执行 resume 100 updates；
- checkpoint、optimizer state 和 resume 链路可生成并检查；
- `metrics=none`，不运行正式 FID/KID。

它不属于官方固定 ECT baseline，不提供训练质量结论。

只读检查环境和资产：

```bash
bash scripts/smoke_engineering_100steps.sh --check-only
```

只执行 `ct_train.py --dry_run`，不训练、不创建 run 目录：

```bash
bash scripts/smoke_engineering_100steps.sh --dry-run
```

脚本默认使用公共基线的规范 AMP 参数：

```text
--fp16=True
--enable_amp=True
```

公共基线也兼容 `--amp=True` 和 `--enable_gradscaler=True`。旧 collaborator smoke
明确使用 `--fp16=False`，且没有启用 GradScaler，因此旧 evidence 不能被解释为
FP16 + GradScaler 验证。只有未来实际运行新的 AMP smoke 并检查训练日志和 state 后，
才能形成新的 GradScaler 运行证据。

完整 100+100-step 工程 smoke 属于训练操作，本次 Day 2 本地整合不会执行。后续只有
得到单独授权后才能运行：

```bash
bash scripts/smoke_engineering_100steps.sh --mode all
```

默认 run 路径为 `/mnt/ect_project/runs/engineering-smoke/`。总 batch 固定为 10，
所以 fresh 1 kimg 为 100 optimizer updates，resume 到 2 kimg 再增加 100 updates。

## 6. 正式固定 ECT baseline

正式 baseline 必须另行定义并冻结以下信息：

- 公共代码 SHA；
- 数据和 checkpoint 内容验证；
- batch、优化器、学习率、seed、schedule 和训练时长；
- AMP/GradScaler 是否实际启用；
- sampling 与 evaluation 协议；
- 正式结果和重复实验。

不能将 `smoke_engineering_100steps.sh` 的结果重命名或描述为正式 baseline。

## 7. 安全检查

提交工程改动前运行：

```bash
bash -n setup_env.sh
bash -n prepare_data.sh
bash -n download_checkpoint.sh
bash -n scripts/smoke_engineering_100steps.sh
python -m compileall scripts training ct_train.py
python ct_train.py --help
git diff --check
git diff --stat
git status
```

数据或 checkpoint 验证失败时应立即停止，不得继续训练。正式训练、完整工程 smoke、
FID 和 KID 都不属于本地整合检查。
