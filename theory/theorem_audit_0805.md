# PR #34：真实 stop-gradient ECT toy 定理审计

日期：2026-08-05

审计对象：[PR #34](https://github.com/hjjjs4vbmv-netizen/recurrence_of_ect/pull/34)，head `799b8ac`

审计范围：推导、实现、测试、CSV、与仓库真实 ECT loss 的一致性，以及相对 ADCM 的 novelty。本审计不修改训练、schedule、controller 或 Role C 的实验产物。

## 结论（指定格式）

**Theorem correctness:** **CONDITIONAL PASS.** `Q_g` 和二阶矩递推在“线性模型、平方 pair loss、target stop-gradient、每步独立重采样 `(z,t)`、固定 gap 与学习率”下正确；但 PR 中“gap ≈ optimizer-step rescaling”尚不是已证明的精确 theorem，只是当前参数化下误差约 `10^-5` 的数值近似。

**ECT specificity:** **PARTIAL.** 推导正确包含共享噪声 pair、target detach、online-branch Jacobian 和 ECT gap；但它没有复现仓库实际训练中的 Pseudo-Huber/范数变换、`1/(t-r)` 权重、网络预条件、minibatch 与非线性参数化。因此应称为“bare linear stop-gradient squared-pair toy”，不能称为真实深网 ECT 的精确动力学。

**Novelty against ADCM:** **DEFENSIBLE AS A DIAGNOSTIC LEMMA, WEAK AS A MAIN THEOREM.** ADCM 用 local/global consistency、JVP 和去噪残差选择离散步长，并未推导优化器的有限时域二阶矩算子 `T_g`；因此 `Q_g/T_g` 分解有不同的分析对象。但本 toy 没有证明 ADCM 的瞬时信息不足，也没有得到 budget-dependent ranking，不能据此宣称 ADCM controller 失败。

**Strongest valid claim:** 在线性高斯、平方 pair loss、stop-gradient toy 中，参数二阶矩由 `T_g=E[Q_g⊗Q_g]` 精确控制，不能把 forward-loss curvature `H_g`、平均更新 `A_g` 与 `T_g` 混为一谈。若进一步满足逐样本更新算子 `R_g(ξ)=a(g)R_1(ξ)`，则取 `η_g=η_1/a(g)` 后 gap 与学习率重标定严格等价；PR #34 的具体 toy 只近似满足该条件。

**Fatal issue, if any:** **对 `Q_g/T_g` 递推本身没有致命错误；对“finite-horizon gap 是 ICLR 理论主线”有致命的负面证据。** A-matched 后没有 crossover，且 `K=500,1000` 仍没有改变 ranking；该 toy 不能支撑独立 gap 机制。必须等待真实深网的 gradient-direction/noise residual，才能决定是否还有主线。

## 1. 可复现性检查

PR #34 的原始测试可通过：

```text
python -m pytest theory/test_true_sg.py -q
4 passed
```

本 PR 另提供独立复算脚本 `theory/audit_scalar_residual.py`。它直接从定义重建 `A_g`、`H_g` 和 `T_g`，不导入被审计的 `true_sg_operator.py`：

```bash
python theory/audit_scalar_residual.py
```

脚本固定 `sample_count=200000`、`seed=0`、`sigma_d=0.5`，生成 `theory/audit_scalar_residual.csv`。CSV 包含完整 `g=0.5:0.05:1.5` 网格上的三个 operator residual，以及 fixed、H-matched、A-matched 在 `K={20,50,100,200,500,1000}` 的 `Tr(M_K)`。原始四个预算的结论为：

| LR mode | `g*`, K=20/50/100/200 | K=200 相对 spread | 结论 |
|---|---:|---:|---|
| fixed | 0.5 / 0.5 / 0.5 / 0.5 | 51.06 | 单调上升 |
| H-matched | 1.5 / 1.5 / 1.5 / 1.5 | 5013.13 | 单调下降 |
| A-matched | 0.5 / 0.5 / 0.5 / 0.5 | `1.38e-5` | 几乎平坦，无 crossover |

原任务要求的 `K={500,1000}` 未进入 PR #34 CSV；审计脚本补算结果为：

| mode | K=500：`g*`, spread | K=1000：`g*`, spread |
|---|---:|---:|
| fixed | 0.5, `5.25e4` | 0.5, `5.24e9` |
| H-matched | 1.5, `4.87e9` | 1.5, `4.48e19` |
| A-matched | 0.5, `1.88e-5` | 0.5, `2.03e-5` |

所以扩展预算后仍无 ranking crossover。A-matched 的 `E_K` 虽对 gap 近乎不变，却从初始约 `2e-4` 增长到 `E_1000≈9.94e-2`；这说明“gap 被重标定消除”和“系统均方稳定”是两件不同的事。

## 2. `Q_g` 是否来自真实 stop-gradient update

Role C 的 toy 定义

```math
f_\beta(x_t,t)=z(1+\beta_1t+\beta_2t^2),\qquad
\ell=\tfrac12(f_t-\operatorname{sg}f_r)^2.
```

令

```math
J_t=[t,t^2]^\top,\qquad
v_g(t)=[t-r,t^2-r^2]^\top.
```

数值 residual 为 `z v_g^Tβ`，autograd 只通过 online branch，因此

```math
\widehat\nabla_{\!\mathrm{sg}}\ell
=z^2J_tv_g^\top\beta,
\qquad
\beta_{k+1}
=\bigl(I-\eta z_k^2J_{t_k}v_g(t_k)^\top\bigr)\beta_k.
```

故 PR #34 的

```math
Q_g(z,t)=I-\eta z^2J_tv_g(t)^\top
```

对该 toy 是正确的。这里的 shared noise 指同一 pair 使用相同 `z`；它不是“精确 PF-ODE pair”。推导没有使用 PF-ODE 数值求解，也不应写成对精确 PF pair 的定理。

## 3. 二阶矩闭合及独立性假设

令 `M_k=E[β_kβ_k^T]`。若每一步的 `(z_k,t_k)` 独立同分布，且独立于由历史样本决定的 `β_k`，则

```math
M_{k+1}=E[Q_{g,k}M_kQ_{g,k}^\top].
```

利用 `E[z²]=σ_d²`、高斯 `E[z⁴]=3σ_d⁴`，得到

```math
M_{k+1}
=M_k-\eta(A_gM_k+M_kA_g^\top)
+3\eta^2\sigma_d^4E_t[(v_g^\top M_kv_g)J_tJ_t^\top],
```

其中 `A_g=σ_d²E_t[J_tv_g^T]`。代码的 3×3 symmetric-basis 实现与该公式一致。

必须在 theorem 中明确的条件：

1. `g`、`η` 和采样分布不依赖当前 `β_k`；自适应 controller 会破坏这个固定算子形式。
2. 每步重采样；若重复使用固定 minibatch、相关 timestep 或跨步共享噪声，不能直接写成 `T_g^K`。
3. update 是齐次线性的，没有 momentum、Adam、weight decay、EMA target drift 或常数 forcing。
4. `t` 在实现中来自一个固定的 200k empirical pool。递推对该离散经验分布是精确的，对目标 clipped-lognormal population 仍有 Monte Carlo quadrature error。
5. 高斯四阶矩 `3σ_d⁴` 是分布特定的；换噪声分布需替换。

### 是否遗漏均值项

没有。`M_k=E[β_kβ_k^T]` 是 raw second moment，不是 covariance。由于 update 为 `β_{k+1}=Q_kβ_k` 且没有加性项，raw moment 已经包含均值贡献，不需要额外添加 `μ_kμ_k^T`。

若改写为 covariance `C_k=M_k-μ_kμ_k^T`，才必须同时递推 `μ_{k+1}=(I-ηA_g)μ_k`，并扣除均值外积。当前文档应始终称 `M_k` 为 second moment，不要称 covariance。

## 4. `H_g`、`A_g`、`T_g` 是否混用

PR #34 的主体推导已正确区分：

| 对象 | 定义 | 控制内容 |
|---|---|---|
| `H_g` | `σ_d²E[v_gv_g^T]` | 数值平方 pair loss 的 curvature |
| `A_g` | `σ_d²E[J_tv_g^T]` | stop-gradient 的平均参数更新；通常非对称 |
| `T_g` | `E[Q_g⊗Q_g]` | raw parameter second moment 的有限时域传播 |

但解释层仍有一个关键跳步：代码只用 Frobenius projection

```math
a(g)=\langle A_g,A_1\rangle_F/\|A_1\|_F^2
```

匹配平均算子。`η_gA_g≈η_1A_1` 并不自动推出 `T_g(η_g)=T_1(η_1)`，因为 `T_g` 还含逐样本算子的二阶项。

`audit_scalar_residual.py` 在完整 `g=0.5:0.05:1.5` 网格上得到：

- `||A_g-aA_1||/||A_g|| ≤ 5.923e-6`；
- 逐样本 operator 的 Frobenius RMS scalar residual `≤5.102e-6`；
- matched 后 `||(T_g-I)-(T_1-I)||/||T_1-I|| ≤1.361e-5`。

这很好地解释了曲线为何平坦，但也证明它只是近似等价。产生非零 residual 的原因包括 `v_g` 第二分量中的 `-Δ²` 和边界 clamp；因此 PR 文案中的“negative theorem”需改为下列两层：

- **可证明 theorem：** 若逐样本 `R_g(ξ)=a(g)R_1(ξ)`，则学习率重标定后 `Q_g(ξ)=Q_1(ξ)`，全部有限时域分布及 moments 完全相同。
- **当前 toy 的结果：** 该条件只近似成立，数值 residual 约 `10^-5`；这是 proposition/observation，不是精确等价 theorem。

## 5. theorem 控制参数误差还是生成误差

当前 `E_K=Tr(M_K)=E||β_K||²` 只控制 toy 的参数二阶矩。它不是 FID、KID、Wasserstein 距离，也没有自动给出深网的生成误差。

若要把它转成该线性 toy 的输出误差，需要另外指定评估分布并证明，例如

```math
E_{t,z}|f_{\beta_K}(x_t,t)-f^*(x_t,t)|^2
=\operatorname{Tr}(G_{\mathrm{eval}}M_K)
```

其中 `G_eval=E[z²φ(t)φ(t)^T]`。PR 当前没有给出这个桥接，更不能外推到深网生成质量。论文中应写“finite-horizon parameter second moment”，不要写“generation error”。

## 6. ECT specificity 审计

该 toy 的 ECT 特征是：相同数据/噪声方向构造 `(t,r)` pair，target branch detach，gap 决定 `r`，online Jacobian 与 pair residual feature 不同。

但仓库真实 `training/loss.py` 还执行：

1. 像素维 squared residual 求和；
2. Pseudo-Huber/平方根变换；
3. 除以 `t-r`；
4. 非线性、预条件深网和 minibatch 优化。

这些操作使真实梯度不再是固定的 `z²J_tv_g^Tβ` 线性形式。因此：

- `Q_g/T_g` theorem 对 bare toy 正确；
- 它不能被命名为“真实 ECT 训练的精确动力学”；
- 对深网可验证的对应物应是 gradient mean、variance、cosine 和 scalar-rescaling residual，而不是直接套用 3×3 `T_g`。

## 7. 与 ADCM 的 novelty 边界

ADCM 的目标不是简单的 loss-only controller。论文式 (8)–(10)同时使用 local consistency、global denoising error、JVP `v` 及其与 denoising residual 的相关性，求解当前网络状态下的离散步长。它还以固定间隔更新 discretization，并报告有限训练预算下的 FID。

因此允许的对比是：

> ADCM 设计的是基于当前网络局部/全局一致性信息的瞬时 discretization rule；本文分析的是给定 gap 下，stop-gradient SGD 的有限时域参数二阶矩传播。ADCM 没有给出该 `T_g` 递推。

不允许的对比是：

- “ADCM 只看当前 loss”；
- “ADCM 只使用 `H_g`”；
- “PR #34 已证明 ADCM 的信息不足”；
- “finite budget 本身足以与 ADCM 区分”。

要证明 instantaneous-statistics insufficiency，仍需构造两个状态：它们对明确列出的 controller observables 完全相同，但有限时域最优 gap 不同。PR #34 没有这个 separation；相反，它得到所有预算下相同的边界 ranking。

## 8. 必须修改的问题

### 写论文前必须修复

1. 将“gap ≈ optimizer-step rescaling theorem”拆成精确的充分条件 theorem 与当前 toy 的近似数值 observation。
2. 标明递推对固定 empirical `t` pool 精确，而非对 lognormal population 解析精确。
3. 把指标统一写成 parameter second moment，删除任何 generation-error 暗示。
4. 补入 `K=500,1000`，并明确报告没有 crossover、长时域均方增长。
5. 将“real stop-gradient ECT toy”改成“bare linear stop-gradient squared-pair ECT toy”。

### PR 文档/复现问题

1. `true_sg_operator.md` 前部仍写“MC <5%, 50k”，后部写“约17%, 200k”；按当前测试重算，vector relative error 约 `14.9%`，trace relative difference 约 `14.2%`。应保留一组定义清楚的数字。
2. 文档写 A-matched spread `~1e-6`，实际 K=200 为 `1.38e-5`；建议写“≤`2.1e-5` for K≤1000”。
3. 文档写 `rho(T) >> 1`，CSV 范围约 `0.99999989–1.022998`；应写“部分 gap 略大于 1，但长期复合导致显著增长”。
4. 文档的 `||A_g||`、`||H_g||` 范围停在旧 grid；当前 `g=0.5–1.5` 实际约为 `337.79–1011.40`、`1.318–11.817`。
5. 文档底部有重复且过时的 Deliverables 段，错误地称 horizon CSV 为 MC 结果。
6. `run(out='.')` 实际把 CSV 写到仓库根目录，而打印信息宣称写入 `theory/`；输出路径与说明不一致。
7. A-flat test 的阈值是 `1e-2`，比文案声称的 `~1e-6` 宽约三阶；应按实际 claim 设为 `1e-4` 左右，并覆盖完整 gap/budget grid。
8. 自带 `_run_all()` 捕获失败后仍以 exit code 0 结束；CI 应直接使用 pytest。

## 9. Go/No-Go 建议

**Toy finite-horizon mainline：NO-GO。** 精确递推成立，但它给出的是负面结果：正确 A-matching 后 gap 影响几乎完全消失，且所有预算的最优点都在同一边界。

**保留为论文 lemma/diagnostic：GO。** `H_g/A_g/T_g` 的区分、精确 raw-second-moment recursion，以及逐样本 scalar-rescaling 的充分条件都可写入 appendix 或 theory motivation。

**是否升级为主线：等待真实深网诊断。** 只有当真实网络中的 normalized gradient noise、方向或分层结构在 scalar matching 后仍有稳定 residual，并且该 residual 能预测 64/128/256 kimg ranking，才值得继续。
