# 最终结论：Fixed sigmoid vs Adaptive v1

## 研究问题与冻结协议

在相同训练 seed、训练预算和采样 NFE 下，Adaptive v1 是否表现出可重复的生成质量优势？比较覆盖 fixed sigmoid / Adaptive v1、训练 seeds 0/1/2、16/32/64 kimg、NFE=1/2（NFE=2 的 `mid_t=0.821`）。每个 checkpoint/NFE 使用 FP32 和固定采样 seeds 0–4999 生成 5,000 个样本。KID-5k 为主指标，FID-5k 为辅助 proxy；二者均越低越好。以下不是标准 FID-50k benchmark。

## 三 seed 配对结果

配对差值定义为 `Adaptive v1 - fixed sigmoid`，负值有利于 Adaptive。`A/F/T` 是 Adaptive 胜 / fixed 胜 / 平局的训练 seed 数。

| Budget | NFE | mean Δ KID ± SD | KID A/F/T | mean Δ FID ± SD | FID A/F/T |
| ---: | ---: | ---: | --- | ---: | --- |
| 16 | 1 | -0.000059 ± 0.008353 | 1/2/0 | -0.951515 ± 4.681037 | 1/2/0 |
| 16 | 2 | +0.001152 ± 0.000535 | 0/3/0 | +0.838204 ± 0.418109 | 0/3/0 |
| 32 | 1 | +0.001110 ± 0.000605 | 0/3/0 | +0.429639 ± 0.312042 | 0/3/0 |
| 32 | 2 | +0.003334 ± 0.003269 | 0/3/0 | +2.221021 ± 1.861612 | 0/3/0 |
| 64 | 1 | +0.000228 ± 0.000741 | 1/2/0 | +0.337120 ± 0.284930 | 0/3/0 |
| 64 | 2 | -0.000854 ± 0.007568 | 1/2/0 | -1.219666 ± 5.743797 | 1/2/0 |

没有任何 budget/NFE 条件达到预先冻结的“至少 2/3 seeds 的主指标优于 fixed”门槛。18 个 KID 配对中 fixed 在 15 个更优；18 个 FID 配对中 fixed 在 16 个更优。16 kimg/NFE=1 和 64 kimg/NFE=2 的有利均值各自由单个 seed 驱动，没有跨 seed 复现；32 kimg 的两个 NFE、两个指标均为 0/3 Adaptive 胜。

## 训练稳定性与 controller

六条连续轨迹均达到 64 kimg，每条 500 attempted iterations；所有记录 loss 有限，NaN/Inf 均为 0，`r/t` 与 gap 始终合法。Fixed 三条轨迹共有 29 个 AMP skipped steps，Adaptive 有 27 个。Adaptive correction 在 3/3 轨迹激活，未出现接近 `max_adjust=0.05` 的饱和步骤；三条轨迹的 correction 符号变化数为 0/2/0。因此 Adaptive v1 能稳定运行并改变 schedule，训练稳定性不比 fixed 差。

## 匿名视觉评价状态

64 kimg 的 24-pair 匿名 A/B 包已由 Role D 生成，但共享服务器上的 ballot 尚未填写；仓库中的 16 kimg 96-pair 包也没有返回的完整 ballot。因此本次数据冻结时没有可报告的匿名 A/B 偏好比例。此项明确记为待完成，不能伪造为 TIE 或据此支持任何方法。

## 最终回答

**负向：当前结果不支持“Adaptive v1 优于 fixed sigmoid”。** 闭环自适应框架在三条训练种子上可运行、会激活且保持稳定，但当前 loss-feedback 控制律没有转化为可重复的生成质量收益；在大多数配对中指标反而略差。由于只有三个训练 seeds、使用 5k proxy 且部分差值远小于 seed 间波动，本结论不应扩展成“fixed 在一般意义上显著更优”，只应表述为：在冻结实现、16/32/64 kimg 和 NFE=1/2 下，没有证据支持 Adaptive v1 的质量优势。

