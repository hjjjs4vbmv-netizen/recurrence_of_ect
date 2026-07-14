# Day 1：工程环境、数据、checkpoint 与断点续训

本文档对应组员 A 的 Day 1 交付。目标是让每个矩池云容器在同一 Git commit 下完成：环境检查、CIFAR-10 准备、官方 EDM 权重准备、100-step 训练、checkpoint 保存和 resume。

## 1. 克隆并记录版本

```bash
git clone https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect.git
cd recurrence_of_ect
git pull --ff-only
git rev-parse HEAD
```

正式 smoke test 默认拒绝存在未提交修改或未跟踪文件的工作区。先把当天工程修改提交到团队分支或主分支，再让其他成员基于该 commit 测试。仅做提交前预检时可给 `smoke_test.sh` 添加 `--allow-dirty`。

## 2. 设置持久化路径（推荐）

矩池云实例重启后，仓库目录和持久化盘的保留策略可能不同。将以下路径替换为本容器实际的持久化目录：

```bash
export ECT_ENV_NAME=ect
export ECT_CIFAR10_TARBALL=/path/to/persistent/cache/cifar-10-python.tar.gz
export ECT_DATA_PATH=/path/to/persistent/datasets/cifar10-32x32.zip
export ECT_TRANSFER_PATH=/path/to/persistent/checkpoints/edm-cifar10-32x32-uncond-vp.pkl
```

未设置时，脚本分别使用仓库下的 `.cache/`、`datasets/` 和 `checkpoints/`；这些目录已被 Git 忽略。

## 3. 创建并检查环境

```bash
bash setup_env.sh
```

脚本使用 `env.yml` 创建 Python 3.9.18、PyTorch 2.3.0、CUDA 12.1 环境，并通过仓库内的 `conda-matpool.yml` 避免矩池云默认配置错误映射 `nvidia` channel。脚本会初始化统一目录，并将实际包版本、GPU 信息和 Git commit 写入：

```text
logs/day1/environment.json
```

已有环境只会被检查，不会自动修改。需要同步 `env.yml` 时运行：

```bash
bash setup_env.sh --update
```

无 GPU 的登录节点只做环境解析时可使用 `--allow-no-cuda`，但这不能算作 Day 1 GPU 验收。

## 4. 准备 CIFAR-10

```bash
bash prepare_data.sh
```

脚本会执行：

1. 下载官方 `cifar-10-python.tar.gz`；
2. 使用官方 MD5 `c58f30108f718f92721af3b95e74349a` 校验下载；
3. 调用仓库的 `dataset_tool.py` 转换为 EDM ZIP；
4. 检查 ZIP CRC、`dataset.json`、图像数量和 32×32 分辨率；
5. 写出 `logs/day1/dataset.json`，其中包含最终数据的 SHA-256。

只检查已准备数据：

```bash
bash prepare_data.sh --check-only
```

## 5. 准备官方 EDM checkpoint

```bash
bash download_checkpoint.sh
```

默认下载仓库训练脚本使用的官方 `edm-cifar10-32x32-uncond-vp.pkl`，检查文件类型和大小，并将 SHA-256 写到：

```text
logs/day1/checkpoint.json
```

只检查已有 checkpoint：

```bash
bash download_checkpoint.sh --check-only
```

## 6. 训练、保存与 resume

先确认当前 commit 已记录且 tracked 文件已提交，然后运行：

```bash
bash smoke_test.sh
```

该命令包含两个阶段：

- Fresh：`total_kimg=1`、总 batch 10，因此正好执行 100 个 optimizer steps；
- Resume：从 Fresh 的 `training-state-000001.pt` 和匹配的网络快照恢复，继续到 `total_kimg=2`，再执行 100 steps。

为了降低显存占用，默认 microbatch 为 2。显存不足时：

```bash
ECT_SMOKE_BATCH_GPU=1 bash smoke_test.sh
```

Smoke test 不计算 FID/KID；其作用是验证训练、日志、样例、网络快照、优化器状态和 resume 链路。输出目录形如：

```text
runs/day1-smoke/<commit>-<UTC时间>/
├── environment.json
├── fresh-100steps/
├── resume-100steps/
└── smoke_report.json
```

`smoke_report.json` 的 `status` 必须为 `passed`。运行目录包含 checkpoint，默认不会上传 GitHub。

## 7. 多容器验收记录

每位复测成员提交以下信息：

```text
成员：
容器/GPU：
Git commit：
environment.json：通过/失败
dataset.json：通过/失败
checkpoint.json：通过/失败
smoke_report.json：通过/失败
运行目录：
备注：
```

Day 1 团队验收至少需要三名成员基于同一 commit 得到 `smoke_report.json: status=passed`。组员 A 负责脚本和故障处理，不能替代其他成员在独立容器中的实际复测。

## 常见问题

- `CUDA is unavailable`：确认容器已分配 GPU，并检查 `nvidia-smi`。
- Conda 包版本不一致：运行 `bash setup_env.sh --update`，不要在个人容器静默安装额外版本。
- 需要使用自定义 Conda 配置：设置 `ECT_CONDA_CONFIG=/path/to/condarc.yml` 后再运行 `setup_env.sh`。
- DDP 端口占用：运行 `bash smoke_test.sh --port 29511`。
- checkpoint 下载中断：保留 `.part` 文件后重跑，脚本会尝试续传。
- 数据或 checkpoint 校验失败：不要继续训练；重新下载，或核对持久化盘文件是否损坏。
