# t→r 映射调度模块说明（Role C）

模块：`training/schedules.py`；测试：`tests/test_schedules.py`。

ECT 训练对 `(x_t, x_r)` 中 `r = r(t, stage)` 由映射调度给出（论文 arXiv 2406.14548
第 3.3 节与附录 A）。本模块把 t→r 调度收敛到统一接口，便于在不改动
`training/loss.py` 的前提下对比官方调度与实验调度。

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
  恒有 `0 <= r < t`。
- `stage`：训练循环维护的课程阶段（官方为 `cur_tick // double_ticks`）；
  仅 `adaptive_v1` 接受小数 stage。
- 超参默认与 `ct_train.py` CLI 一致：`q=2.0, k=8.0, b=1.0`。
- 兼容 `ECMLoss` 的有状态用法：`schedule.update_schedule(stage)` 后调
  `schedule.t_to_r(t)`。

## 支持的调度

| 名称 | 公式（`decay = 1/q^(stage+1)`，均 clamp `r>=0`） | 来源 |
| :-- | :-- | :-- |
| `const` | `r/t = 1 - decay` | 官方 Eq.(17)，`ECMLoss.t_to_r_const` 原样移植 |
| `sigmoid` | `r/t = 1 - decay * n(t)`，`n(t) = 1 + k*sigmoid(-b*t)` | 官方 Eq.(18)，训练默认，`ECMLoss.t_to_r_sigmoid` 原样移植 |
| `adaptive_v1` | 同 `sigmoid`，但 `stage` 允许小数 | Role C 实验 v1 |

**官方 fixed 公式不变的保证**：`const` / `sigmoid` 是 `training/loss.py` 公式的
逐句移植（`training/loss.py` 本身未被改动）；
`tests/test_schedules.py::OfficialFormulaParityTest` 对多组 `(q,k,b,stage,dtype)`
与 `ECMLoss.t_to_r` 做**按位相等**校验，两边任何一处被改动测试都会失败。

## adaptive_v1 设计（v1，待评审）

- **动机**：官方每 `double_ticks` 个 tick 令 `decay` 突降为 `1/q`
  倍，`Δt = t - r` 随之跳变，训练损失尺度在阶段边界出现阶跃。
- **定义**：沿用 sigmoid 公式，把 `stage` 换成连续训练进度
  `stage = cur_tick / double_ticks`（模块提供 `continuous_stage()` 帮助函数），
  使 `Δt` 随进度几何平滑收缩。
- **性质**（均有测试）：整数 stage 处与官方 `sigmoid` **按位一致**（每个阶段
  起点锚定 baseline）；小数 stage 的 `r` 落在相邻两个整数 stage 之间；`r`
  随进度单调收紧；不引入任何新超参。
- **接入**：尚未接入 `ct_training_loop.py`（Day2 审计的 protected 文件）。
  计划的改动为一行——阶段更新处把 `cur_tick // double_ticks` 换成
  `continuous_stage(cur_tick, double_ticks)` 并每个 tick 调用
  `loss_fn.update_schedule(...)`——待队内评审后单独提交。

## 现有 run 配置下的 stage 行为（供对照）

- `run_ecm.sh`：`tick=12.8, duration=25.6, double=250` → 共 2000 tick、
  stage 0..7，最终 `decay = 2^-8 = 1/256`。
- `run_ecm_1hour.sh`：`-q 256 --double 10000` → 全程 stage 0，
  `decay = 1/256`（直接锁定终点紧度）。

## 验证

```bash
python -m unittest tests.test_schedules -v   # 16 个用例
python -m training.schedules                 # 打印三种调度的 r/t 表
```
