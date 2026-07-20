# t→r 映射调度模块说明（Role C）

模块：`training/schedules.py`；测试：`tests/test_schedules.py`。

ECT 训练对 `(x_t, x_r)` 中 `r = r(t, stage)` 由映射调度给出（论文 arXiv 2406.14548
第 3.3 节与附录 A）。本模块把 t→r 调度收敛到统一接口；`ECMLoss` 的 t→r 入口
已改为经由本模块派发（`__init__` 中 `get_schedule(adj, q=q, k=k, b=b)`，
`__call__` 中 `r = self.schedule.compute_r(t=t, stage=self.stage)`），
`training/loss.py` 中除该入口外的训练逻辑（ε 共享、teacher `no_grad`、
dropout RNG 保存/恢复、损失权重、Huber 参数）逐字节未动。

## 接口

两种调用形式等价：

```python
from training.schedules import compute_r, get_schedule

# 对象式
schedule = get_schedule("sigmoid", q=256, k=8, b=1)
r = schedule.compute_r(t=t, stage=stage)

# 函数式
r = compute_r(t=t, stage=stage, schedule="sigmoid", q=256, k=8, b=1)
```

- `t`：任意形状的 torch 张量（训练循环里是 `[N,1,1,1]`），也接受
  python/numpy 标量或数组（内部 `torch.as_tensor` 转换）；返回同形状张量，
  恒有 `0 <= r <= t`。
- `stage`：训练循环维护的课程阶段；三种调度均保持官方
  `cur_tick // double_ticks` 的整数 stage。`adaptive_v1` 只根据 loss EMA
  修正 `r/t`，不另外改变 stage 课程。
- 超参默认与 `ct_train.py` CLI 一致：`q=2.0, k=8.0, b=1.0`。
- 兼容 `ECMLoss` 的有状态用法：`schedule.update_schedule(stage)` 后调
  `schedule.t_to_r(t)`。

## 支持的调度

| 名称 | 公式（`decay = 1/q^(stage+1)`，均 clamp `r>=0`） | 来源 |
| :-- | :-- | :-- |
| `const` | `r/t = 1 - decay` | 官方 Eq.(17)，`ECMLoss.t_to_r_const` 原样移植 |
| `sigmoid` | `r/t = 1 - decay * n(t)`，`n(t) = 1 + k*sigmoid(-b*t)` | 官方 Eq.(18)，训练默认，`ECMLoss.t_to_r_sigmoid` 原样移植 |
| `adaptive_v1` | 官方 `sigmoid` ratio + loss EMA 驱动的有界修正 | Role C 实验 v1 |

**官方 fixed 公式不变的保证**：官方公式方法 `t_to_r_const` / `t_to_r_sigmoid`
**原样保留**在 `training/loss.py` 中作为 parity 基准（训练路径不再调用它们，
仅供测试对照）；`tests/test_schedules.py::OfficialFormulaParityTest` 对多组
`(q,k,b,stage,dtype)`（A100 上还含 cpu/cuda 两种设备）把 schedules 模块输出与
这两个参考方法做**按位相等**校验，`ECMLossIntegrationTest` 再校验接入后的
`self.schedule.compute_r` 入口与参考方法按位一致。任何一侧公式被改动，测试
都会失败。

## adaptive_v1 设计

设官方 sigmoid 比率为 `rho_0 = r/t`，首个有限 loss EMA 为 `L_0`，
当前 loss EMA 为 `L_ema`：

```text
score = tanh(log(L_0) - log(L_ema))
delta = adaptive_max_adjust * score
rho   = clamp(rho_0 + delta, 0, 1 - adaptive_min_gap)
r     = t * rho
```

`adaptive_v1` 在没有有效 correction（尚无有效 signal、warmup 中，或
`adaptive_max_adjust=0`）时直接返回官方 sigmoid 的 `rho_0`，按位保持 fixed
baseline。只有 correction 激活后才应用 `adaptive_min_gap` 上限与上式中的修正。

- loss 下降时 `delta > 0`，减小 `t-r`，加强一致性约束；
- loss 恶化时 `delta < 0`，增大 `t-r`，降低当前任务难度；
- `|delta| <= adaptive_max_adjust`，不做搜索或额外控制器；
- 非有限、负数 loss 信号直接忽略；输出做有限化和边界 clamp，
  保证 `0 <= r <= t`且不产生 NaN/Inf；
- loss 信号每 `adaptive_update_kimg`（默认 0.5 kimg）按**绝对图像数边界**
  聚合；它在训练迭代内执行，不依赖 `--tick`/maintenance（默认 50 kimg）。
  每个窗口的 sum/count 以 all-reduce 合并，因此同样输入与状态下各 rank 使用
  同一修正值。
- 前 `adaptive_warmup_updates` 个有效聚合窗口只建立 loss EMA；warmup 完成后的
  下一次更新才允许非零修正。

默认参数只是首版单点配置，不代表完成大范围搜索：

| CLI | 默认值 | 含义 |
| :-- | --: | :-- |
| `--adaptive-loss-ema-beta` | `0.9` | loss EMA 平滑系数 |
| `--adaptive-update-kimg` | `0.5` | 触发一次全局 adaptive loss 聚合的图像间隔；与 `--tick` 独立 |
| `--adaptive-warmup-updates` | `2` | 应用修正前仅更新 EMA 的有效信号窗口数 |
| `--adaptive-max-adjust` | `0.05` | `r/t` 最大绝对修正 |
| `--adaptive-min-gap` | `0.001` | 开启修正时的最小 `(t-r)/t` |

### 稳定运行时 telemetry 接口

训练与结果收集代码只通过 `loss_fn.schedule_runtime_metrics()` 读取调度状态，
不访问 schedule 的内部字段。该接口固定返回：

- `loss_ema`、`loss_reference`、`correction`、`signal_updates`；
- `adaptive_active`：warmup 完成且 correction 控制器已激活；
- `r_over_t_mean`：最近实际训练 pair 的 `mean(r/t)`；
- `gap_mean`：最近实际训练 pair 的 `mean((t-r)/t)`。

`train_summary.csv` 每个 attempted iteration 记录这些字段。controller 状态是在
本 iteration 末尾（可能完成 signal update 后）读取，`r_over_t_mean` 与
`gap_mean` 则描述本 iteration 实际使用的 pair；因此 correction 的变化应在后续
iteration 的 pair 指标中体现。

从旧的 11 列 `train_summary.csv` 续训时，只接受完全匹配的旧标准表头。训练循环
会先保存 `.pre-telemetry.bak`，再原子迁移到新表头；无法重建的历史 telemetry
保持空值。collector 允许这一段连续的历史空前缀，并在 metadata 中记录覆盖率与
首个 telemetry iteration；telemetry 开始后的空洞、部分表头或非法状态都会拒绝。

adaptive 运行状态会随 training-state 保存/恢复；完整参数与当前 EMA、
参考 loss、修正值会写入 checkpoint 中的 `loss_fn`，并由固定种子评估
metadata 记录为 `training_schedule`。

### CLI 兼容性

```bash
# 原命令：语义不变
python ct_train.py ...
python ct_train.py ... --mapping sigmoid
python ct_train.py ... --mapping const

# 新入口
python ct_train.py ... --schedule adaptive_v1
# 同时接受连字符写法 adaptive-v1，内部统一记录为 adaptive_v1

# 显式关闭，严格恢复官方 fixed sigmoid
python ct_train.py ... --schedule sigmoid
```

`--schedule` 与 `--mapping` 指向同一个内部 `mapping` 字段，因此旧的配置
传递、`loss_kwargs.adj` 和日志结构不会因参数改名而变化。不传
两者时仍默认 `sigmoid`，不会创建 adaptive loss Collector 或改变官方
stage 边界。

## 现有 run 配置下的 stage 行为（供对照）

- `run_ecm.sh`：`tick=12.8, duration=25.6, double=250` → 共 2000 tick、
  stage 0..7，最终 `decay = 2^-8 = 1/256`。
- `run_ecm_1hour.sh`：`-q 256 --double 10000` → 全程 stage 0，
  `decay = 1/256`（直接锁定终点紧度）。

## 验证

```bash
python -m unittest tests.test_schedules tests.test_training_cli_compat -v
# 其中 2 个 ECMLoss.__call__ 端到端用例需 CUDA（A100 激活）
python -m training.schedules                 # 打印三种调度的 r/t 表
```
