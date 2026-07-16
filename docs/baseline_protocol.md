# 基线协议（260716）

## 决策

| 项 | 选择 | 理由 |
|------|--------|-----------|
| 数据集 | CIFAR-10 32×32 | 官方 ECT `main` 分支目标数据集 |
| 条件控制 | **无条件**（`--cond=0`） | 与 `run_ecm*.sh` 及 EDM 迁移 pickle `edm-cifar10-32x32-uncond-vp.pkl` 一致 |
| 教师模型 | 官方 EDM VP 无条件模型 | 见 `checkpoint_manifest.json` |
| 数据格式 | EDM ZIP+PNG | 通过 `dataset_tool.py` 准备 |

## 环境

```bash
conda env create -f env.yml
conda activate ect
# 为兼容 diffusers 0.26.3 做版本锁定
pip install 'huggingface_hub<0.26'
```

## 数据

```bash
mkdir -p datasets
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P datasets/
python dataset_tool.py --source=datasets/cifar-10-python.tar.gz --dest=datasets/cifar10-32x32.zip
```

## 检查点

见 `checkpoint_manifest.json`。本地/缓存中的教师模型：

- 缓存路径：`/root/.cache/dnnlib/downloads/c320a0e2338e26e7ce763402b5b56d98_https___nvlabs-fi-cdn.nvidia.com_edm_pretrained_edm-cifar10-32x32-uncond-vp.pkl`
- MD5 `0cb62d16b8617fd703a697cc17ae422e`
- SHA256 `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`
- 大小约 212.83 MiB

## 固化配置

- `configs/baselines/cifar10_uncond_ect_1hour.json`
- `configs/baselines/cifar10_uncond_ect_200k.json`

## 第 1 天验证

```bash
python /tmp/day1_verify.py
# 或
python /root/ect_day1_deliverables/tools/day1_verify.py
```

### 计算图断言（全部通过）

1. `r < t`
2. `r >= 0`
3. `x_t = x_0 + t ε`
4. `x_r = x_0 + r ε`
5. 共享 `ε`
6. 低噪声分支 `D_yr` 在 `torch.no_grad()` 下
7. 通过 CUDA RNG 的保存/恢复实现共享 dropout mask

### 冒烟测试 + 断点续训（已通过）

- 约 100 次优化器步（13 kimg，batch 128），EDM 迁移
- 检查点保存并续训至 tick 14
- 冒烟测试目录：`/tmp/ect_day1_smoke/`

### 交付物

| 交付物 | 位置 |
|-------------|----------|
| `baseline_protocol.md` | 本文件 / 仓库根目录（若已同步） |
| `checkpoint_manifest.json` | 仓库根目录 |
| `configs/baselines/` | 仓库内 |
| `logs/smoke_test/` | `/root/ect_day1_deliverables/logs/smoke_test/`（**`/mnt` 配额已满，未同步**） |
| `samples/edm_seed_grid.png` | `/root/ect_day1_deliverables/samples/edm_seed_grid.png` |

## 关于 /mnt 5G 配额的说明

第 1 天导出时 `/mnt` 触发 Disk quota exceeded。大型冒烟检查点位于 `/tmp/ect_day1_smoke`。若要将日志/样本同步到仓库，请先腾出空间（删除 `ct-runs/day1-smoke/` 中的失败导出以及多余的 `training-state-*.pt`），然后执行：

```bash
cp /root/ect_day1_deliverables/logs/smoke_test/* /mnt/recurrence_of_ect/logs/smoke_test/
cp /root/ect_day1_deliverables/samples/edm_seed_grid.png /mnt/recurrence_of_ect/samples/
cp /root/ect_day1_deliverables/tools/day1_verify.py /mnt/recurrence_of_ect/tools/
cp /root/ect_day1_deliverables/baseline_protocol.md /mnt/recurrence_of_ect/
```
