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
- `stage`：训练循环维护的课程阶段；四种调度均保持官方
  `cur_tick // double_ticks` 的整数 stage。实验调度只根据 loss EMA
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
| `pid_deadband` | 以带 deadband 的有界 PID 指数缩放官方 `sigmoid` gap | Idea 7 实验 |

**官方 fixed 公式不变的保证**：官方公式方法 `t_to_r_const` / `t_to_r_sigmoid`
**原样保留**在 `training/loss.py` 中作为 parity 基准（训练路径不再调用它们，
仅供测试对照）；`tests/test_schedules.py::OfficialFormulaParityTest` 对多组
`(q,k,b,stage,dtype)`（A100 上还含 cpu/cuda 两种设备）把 schedules 模块输出与
这两个参考方法做**按位相等**校验，`ECMLossIntegrationTest` 再校验接入后的
`self.schedule.compute_r` 入口与参考方法按位一致。任何一侧公式被改动，测试
都会失败。

## adaptive_v1 设计

设官方 sigmoid 比率为 `rho_0 = r/t`，warm-up 最后一次更新后的 loss EMA
为 `L_ref`，当前 loss EMA 为 `L_ema`：

```text
score = tanh(log(L_ref) - log(L_ema))
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
- 前 `adaptive_warmup_updates` 个有效聚合窗口只建立 loss EMA；最后一个
  warm-up 窗口更新 EMA 后，将该 EMA 固定为 `loss_reference`。下一次更新才
  允许非零修正。

默认参数只是首版单点配置，不代表完成大范围搜索：

| CLI | 默认值 | 含义 |
| :-- | --: | :-- |
| `--adaptive-loss-ema-beta` | `0.9` | loss EMA 平滑系数 |
| `--adaptive-update-kimg` | `0.5` | 触发一次全局 adaptive loss 聚合的图像间隔；与 `--tick` 独立 |
| `--adaptive-warmup-updates` | `2` | 应用修正前仅更新 EMA 的有效信号窗口数 |
| `--adaptive-max-adjust` | `0.05` | `r/t` 最大绝对修正 |
| `--adaptive-min-gap` | `0.001` | 开启修正时的最小 `(t-r)/t` |

## pid_deadband 设计

`pid_deadband` 复用相同的全局 loss 聚合与 EMA 通路。warm-up 结束时冻结
`L_ref`，之后以对数相对误差作为 PID 输入：

```text
e_k = log(L_ref) - log(L_ema)

if abs(e_k) < epsilon:
    u_k = 0                         # deadband，同时冻结积分项
else:
    I_k = clamp(I_(k-1) + e_k, -I_max, I_max)
    u_k = clamp(Kp*e_k + Ki*I_k + Kd*(e_k-e_(k-1)), -u_max, u_max)

gap_0 = (t-r_sigmoid)/t
gap_k = gap_0                                      if u_k == 0
        clamp(gap_0 * exp(-u_k), adaptive_min_gap, 1) otherwise
r_k   = t * (1-gap_k)
```

- loss 好于参考值时 `e_k > 0`，`exp(-u_k) < 1`，pair 变紧；loss 恶化时反向；
- deadband 内 `u_k` 严格为 0，积分冻结，但会更新上一误差以避免离开 deadband
  时出现 derivative kick；
- `u_k=0` 时严格复用官方 gap，`adaptive_min_gap` 不介入，因而满足
  `g_k=g_0*exp(0)=g_0`；
- 除积分硬限幅外还使用 conditional integration：若输出已饱和且当前误差继续把
  输出推向饱和方向，则拒绝该次积分，避免 wind-up；
- controller 激活前直接调用官方 sigmoid 实现，因此 fixed 路径保持按位一致；
- `correction` telemetry 对该调度表示 PID 输出 `u_k`，实际 gap 缩放量为
  `exp(-u_k)`；PID 的完整状态随 training-state 和 snapshot 保存/恢复。

PID 默认参数采用保守单点配置，尚未经过远程训练调参：

| CLI | 默认值 | 含义 |
| :-- | --: | :-- |
| `--pid-update-kimg` | `51.2` | PID 信号更新周期；25.6 MIMG 训练恰好 500 次更新 |
| `--pid-kp` | `0.1` | 比例增益 |
| `--pid-ki` | `0.01` | 积分增益 |
| `--pid-kd` | `0.05` | 微分增益 |
| `--pid-deadband` | `0.02` | 对数相对误差 deadband（约 2%） |
| `--pid-integral-limit` | `5.0` | 积分项绝对限幅 |
| `--pid-max-control` | `0.1` | `u_k` 绝对限幅；gap scale 约限制在 `[0.905, 1.105]` |
| `--pid-lr-boost` | `1.25` | PID 分支相对 `--lr` 的基础学习率提升 |
| `--pid-lr-max-boost` | `1.5` | 有效学习率倍率硬上限 |
| `--pid-lr-warmup-kimg` | `256` | 从 `1.0x` 平滑升至目标倍率的图像预算 |

同时复用 `--adaptive-loss-ema-beta`、`--adaptive-warmup-updates` 与
`--adaptive-min-gap`。如短程 smoke run 需要观察 PID 激活，应显式减小
`--pid-update-kimg`；正式对比则建议按 `total_kimg / 目标更新次数` 选取周期，
不要把参数搜索结果与最终评估使用同一 seed。

### PID 学习率扩展

学习率扩展只对 `pid_deadband` 生效，`const`、`sigmoid` 和 `adaptive_v1`
始终保持用户传入的 `--lr`。每个 optimizer step 使用：

```text
target_multiplier = clamp(pid_lr_boost * exp(u_k), 1, pid_lr_max_boost)
warmup_alpha       = clamp(processed_kimg / pid_lr_warmup_kimg, 0, 1)
lr_multiplier     = 1 + warmup_alpha * (target_multiplier - 1)
effective_lr      = base_lr * lr_multiplier
```

默认 `u_k∈[-0.1,0.1]` 时，warm-up 结束后的目标倍率约为
`[1.13,1.38]`，并始终限制在 `[1.0,1.5]`。倍率由 checkpoint 中已有的
`cur_nimg` 和 PID 状态确定，不需新增可变状态，因此 resume 与不间断训练一致。
实际学习率写入 `Progress/learning_rate`，倍率写入
`Schedule/pid_lr_multiplier_used`；基于本 iteration 结束后最新控制量计算的
下一步倍率写入 `Schedule/pid_lr_multiplier_next`。PID-LR 配置同时写入
training-state，resume 时若基础 LR、boost、上限或 warm-up 不一致会拒绝续训，
避免静默改变学习率轨迹。

### 稳定运行时 telemetry 接口

训练与结果收集代码只通过 `loss_fn.schedule_runtime_metrics()` 读取调度状态，
不访问 schedule 的内部字段。该接口固定返回：

- `loss_ema`、`loss_reference`、`correction`、`signal_updates`；
- `adaptive_active`：warmup 完成且 correction 控制器已激活；
- `r_over_t_mean`：最近实际训练 pair 的 `mean(r/t)`；
- `gap_mean`：最近实际训练 pair 的 `mean((t-r)/t)`。

`pid_deadband` 还通过同一接口向 `stats.jsonl` 报告 `pid_error`、
`pid_integral`、`pid_derivative` 和 `pid_deadband_active`，用于远程检查 P/I/D
各项与 deadband 是否按预期工作；通用的 `train_summary.csv` schema 保持不变。

`train_summary.csv` 每个 attempted iteration 记录这些字段，以及
`next_loop_cur_tick`。后者是本 iteration 完成后下一循环将使用的真实 tick；若该次
maintenance 保存 checkpoint，它与 checkpoint 中的 `cur_tick` 完全一致，因此
collector 不会从 `processed_nimg` 或 `--tick` 反推 tick。controller 状态是在
本 iteration 末尾（可能完成 signal update 后）读取，`r_over_t_mean` 与
`gap_mean` 则描述本 iteration 实际使用的 pair；因此 correction 的变化应在后续
iteration 的 pair 指标中体现。

collector 的 activation gate 将两个 iteration 明确区分：
`first_nonzero_correction_iteration` 表示本 iteration 结束后 controller
第一次得到非零 correction；`first_adapted_pair_iteration` 只能是其下一次
iteration，因为该次 pair 才会把 correction 用于 `r(t)`。对于
`--mode activation --schedule adaptive_v1`，若最终 `signal_updates < 3`、
最终 `adaptive_active` 不为真、没有非零 correction、下一实际 pair 不在结束前，
或 correction 后少于 4 个 attempted iterations，collector 会拒绝打包。

从旧的 11 列 `train_summary.csv` 续训时，只接受完全匹配的旧标准表头。训练循环
会先保存 `.pre-telemetry.bak`，再原子迁移到新表头；无法重建的历史 telemetry
保持空值。collector 允许这一段连续的历史空前缀，并在 metadata 中记录覆盖率与
首个 telemetry iteration；telemetry 开始后的空洞、部分表头或非法状态都会拒绝。

adaptive 运行状态会随 training-state 保存/恢复；完整参数与当前 EMA、
参考 loss、修正值会写入 checkpoint 中的 `loss_fn`。后续将由 Role D 的独立
follow-up 把 `training_schedule` 接入固定种子评估 metadata。

### CLI 兼容性

```bash
# 原命令：语义不变
python ct_train.py ...
python ct_train.py ... --mapping sigmoid
python ct_train.py ... --mapping const

# 新入口
python ct_train.py ... --schedule adaptive_v1
# 同时接受连字符写法 adaptive-v1，内部统一记录为 adaptive_v1

# 带 deadband 的 PID；也接受 pid-deadband
python ct_train.py ... --schedule pid_deadband \
  --pid-update-kimg 51.2 --pid-kp 0.1 --pid-ki 0.01 --pid-kd 0.05 \
  --pid-deadband 0.02 --pid-integral-limit 5 --pid-max-control 0.1 \
  --pid-lr-boost 1.25 --pid-lr-max-boost 1.5 --pid-lr-warmup-kimg 256

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
python -m training.schedules                 # 打印四种调度的 r/t 表
```
