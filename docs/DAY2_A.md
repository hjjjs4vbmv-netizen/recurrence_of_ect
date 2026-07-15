# Day 2：通用训练、自动恢复与结果导出

本文档对应组员 A 的 Day 2 工程交付。目标是让 B、C 使用同一个启动入口运行基线和自适应实验，让中断后的任务从完整 checkpoint 恢复，并让 D 收到完全一致的精简结果包。

## 1. Day 2 提供了什么

- `train.sh`：读取版本化配置，检查 Git 状态，记录运行元数据并启动训练；
- `scripts/launch_training.py`：选择完整 checkpoint 对、生成 `torchrun` 命令并记录退出状态；
- `configs/cifar10_a100.template.yaml`：单卡 A100 配置模板，不包含个人路径；
- `checkpoint-latest.json`：记录最新 state/snapshot 对的 tick、大小和训练进度；
- `export_results.sh`：将一次运行整理成团队约定的七文件结果包；
- `tests/test_day2_tools.py`：覆盖环境变量、断点选择和结果导出。

配置文件采用 JSON-compatible YAML。它是合法的 YAML 1.2，同时能由 Python 标准库读取，因此不需要为了配置解析再安装依赖。

## 2. 正式实验前的路径配置

数据、官方 checkpoint 和大型训练产物应放在矩池云持久化目录，不放入 GitHub。每个实验使用独立目录：

```bash
export ECT_ENV_NAME=ect
export ECT_DATA_PATH=/mnt/recurrence_of_ect/day2/datasets/cifar10-32x32.zip
export ECT_TRANSFER_PATH=/mnt/recurrence_of_ect/day2/checkpoints/edm-cifar10-32x32-uncond-vp.pkl
export ECT_RUN_DIR=/mnt/recurrence_of_ect/day2/runs/<方法>-<seed>-<日期>
```

先让实验 owner 复制模板并只修改本实验需要的参数：

```bash
cp configs/cifar10_a100.template.yaml configs/<实验名>.yaml
```

必须填写 `experiment.name`、`owner` 和 `purpose`。B 负责基线配置，C 负责自适应调度配置；A 只维护入口和格式，不替实验 owner 决定 policy。

## 3. 预检与正式启动

提交代码后先打印实际命令，不占用 GPU：

```bash
bash train.sh --config configs/<实验名>.yaml --dry-run
```

正式运行：

```bash
bash train.sh --config configs/<实验名>.yaml
```

正式运行默认要求：

- 当前 tracked/untracked 修改已经提交；
- 数据 ZIP 存在；
- 新运行存在 transfer checkpoint，或旧运行存在完整 resume checkpoint；
- 配置中的 GPU 数量和 DDP 端口有效。

仅在提交前做短预检时允许使用 `--allow-dirty`。该选项产生的数字不能进入最终对比表。

## 4. 自动保存与断点恢复

训练保存 checkpoint 时先写临时文件，再原子替换最终文件，避免中断时留下“文件名存在但内容只写了一半”的断点。最新 checkpoint 完成后写出：

```text
checkpoint-latest.json
network-snapshot-latest.pkl
training-state-latest.pt
```

相同命令重新运行时，`auto_resume=true` 会：

1. 检查 `checkpoint-latest.json` 指向的 state/snapshot；
2. 同时检查完整编号 checkpoint 对，选择 tick 最高的一组，同 tick 时优先清单；
3. 如果目录已有训练输出但没有完整 checkpoint，则停止并报告，避免静默覆盖；
4. 如果是全新目录，则从配置中的 transfer checkpoint 开始。

手动指定断点：

```bash
bash train.sh \
  --config configs/<实验名>.yaml \
  --resume /mnt/.../training-state-000020.pt
```

强制全新运行只能针对空目录：

```bash
bash train.sh --config configs/<实验名>.yaml --resume none
```

每次启动会写 `run_metadata.json`，每次开始和结束事件会追加到 `launcher_history.jsonl`。其中包含 commit、分支、是否 dirty、GPU、配置哈希、完整命令、resume 来源和退出码。

## 5. 给 D 的统一结果包

训练结束或需要快速对比时运行：

```bash
bash export_results.sh \
  --run-dir "${ECT_RUN_DIR}" \
  --output-dir artifacts/day2/<实验名>
```

输出固定为：

```text
config.yaml
metadata.json
metrics.csv
train_summary.csv
samples_64.png
schedule_curve.png
notes.md
```

- `config.yaml` 同时保存提交配置和训练程序最终解析的配置；
- `metadata.json` 保存运行元数据和 checkpoint 清单，但不复制大型 checkpoint；
- `metrics.csv` 汇总所有 `metric-*.jsonl`；训练阶段关闭正式指标时只有表头，这是正常情况；
- `train_summary.csv` 提取 loss、时间、显存、schedule stage/ratio；
- `samples_64.png` 固定导出前 8×8、共 64 张样例；
- `schedule_curve.png` 从 `stats.jsonl` 绘制，旧日志则尝试从 `log.txt` 回退解析；
- `notes.md` 必须由实验 owner 补上观察、异常和下一步决定。

未完成的短跑只用于排错，可以临时添加 `--allow-incomplete` 生成明显标注的占位文件；这种结果包不能进入正式表格。

## 6. 存储规则

上传 GitHub：配置、元数据、CSV、64 张样例、曲线和 notes。

保留在 `/mnt`：`training-state-*.pt`、`network-snapshot-*.pkl`、完整样本、数据集和下载缓存。

空间不足时优先保留最终完整 state/snapshot 对和七文件结果包；已经能由脚本重新下载的数据集、官方 checkpoint、旧的重复 checkpoint 可清理。删除前必须确认最终 checkpoint 对可以被 `train.sh --dry-run` 识别。

## 7. 本地检查

以下检查不需要 CUDA：

```bash
bash -n train.sh export_results.sh
python -m unittest discover -s tests -v
python -m py_compile ct_train.py training/ct_training_loop.py scripts/*.py
```

GPU 验收仍需在矩池云执行一次“启动—产生 checkpoint—中断—同命令恢复—导出结果包”的完整流程，并把 commit、运行目录和导出目录记录到当天看板。
