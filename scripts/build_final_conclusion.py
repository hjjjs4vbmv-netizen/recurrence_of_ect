#!/usr/bin/env python3
"""Build the one-page decision record from locked quantitative, blind, and stability results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"[build_final_conclusion] ERROR: {message}")


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")


def load_csv(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


def direction(item: dict) -> str:
    mean_delta = float(item["mean_delta"])
    adaptive, fixed, _tie = item["adaptive_fixed_tie_seed_counts"]
    if mean_delta < 0 and adaptive >= 2:
        return "adaptive"
    if mean_delta > 0 and fixed >= 2:
        return "fixed"
    return "mixed"


def verdict_text(directions: list[str], stable: bool) -> str:
    if directions == ["adaptive", "adaptive"] and stable:
        return "结果在两个 NFE 条件下均方向性支持 Adaptive v1，但证据仅限三训练种子与 5k proxy，不能表述为标准基准上的确定优势。"
    if directions == ["fixed", "fixed"]:
        return "结果不支持“Adaptive v1 优于 fixed sigmoid”；两个 NFE 条件均方向性偏向 fixed sigmoid。"
    return "结果为混合或持平，不能支持“Adaptive v1 在相同预算与 NFE 下总体优于 fixed sigmoid”。"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantitative-dir", type=Path, required=True)
    parser.add_argument("--blind-dir", type=Path, required=True)
    parser.add_argument("--stability-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    quantitative = load_json(args.quantitative_dir / "quantitative_summary.json")
    metric_rows = load_csv(args.quantitative_dir / "quantitative_metrics.csv")
    blind = load_json(args.blind_dir / "blind_ab_summary.json")
    stability = load_json(args.stability_dir / "training_stability.json")

    primary = quantitative["primary_metric"]
    primary_label = "KID-5k" if primary == "kid5k_full" else "FID-5k"
    nfe_items = [quantitative["summary_by_nfe"][str(nfe)][primary] for nfe in (1, 2)]
    directions = [direction(item) for item in nfe_items]
    stable = bool(stability.get("all_six_runs_complete") and stability.get("all_losses_finite"))
    verdict = verdict_text(directions, stable)
    overall_blind = next(row for row in blind["summary"] if row["stratum"] == "overall")

    means = {}
    for nfe in (1, 2):
        means[nfe] = {}
        for schedule in ("sigmoid", "adaptive_v1"):
            values = [
                float(row[primary])
                for row in metric_rows
                if int(row["nfe"]) == nfe and row["schedule"] == schedule
            ]
            means[nfe][schedule] = statistics.mean(values)

    lines = [
        "# 最终结论：Fixed sigmoid vs Adaptive v1（16 kimg）",
        "",
        "## 研究问题",
        "",
        "在相同 16 kimg 训练预算和相同 NFE 下，Adaptive v1 是否优于 fixed sigmoid？",
        "",
        "## 冻结设计",
        "",
        "CIFAR-10 32×32；fixed sigmoid 与 Adaptive v1；训练 seeds 0/1/2；NFE=1 与 NFE=2（mid_t=0.821）；FP32；每个 checkpoint/NFE 使用相同逐样本 seeds 0–4999。主指标为 KID-5k；若事先规定的 45 分钟可运行性门槛触发，则以 FID-5k + 盲评降级。所有数值均为 5k-sample proxy，不是标准 FID-50k benchmark。",
        "",
        "## 定量结果（越低越好）",
        "",
        f"| NFE | Fixed 三-seed 均值 {primary_label} | Adaptive 三-seed 均值 {primary_label} | 配对均值 Δ（A−F） | 配对 SD | A/F/T seeds |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for nfe, item in zip((1, 2), nfe_items):
        lines.append(
            f"| {nfe} | {means[nfe]['sigmoid']:.6f} | {means[nfe]['adaptive_v1']:.6f} | "
            f"{item['mean_delta']:.6f} | {item['sample_sd_delta']:.6f} | {item['adaptive_fixed_tie_seed_counts']} |"
        )
    lines.extend([
        "",
        "## 匿名 A/B 与稳定性",
        "",
        f"盲评包含 {blind['complete_raters']} 名完整评审者、每人 96 个配对 trial：Adaptive/Fixed/Tie = "
        f"{overall_blind['adaptive_v1']}/{overall_blind['sigmoid']}/{overall_blind['tie']}；"
        f"ties=0.5 的 Adaptive 得分为 {overall_blind['adaptive_tie_half_score']:.3f}。",
        "",
        f"六个训练均完成 16 kimg：{stability.get('all_six_runs_complete')}；全部记录 loss 有限：{stability.get('all_losses_finite')}。"
        f"Adaptive controller 在 {stability['summary_by_schedule']['adaptive_v1'].get('controller_activated_runs', 0)}/3 个 run 中激活。",
        "",
        "## 回答",
        "",
        verdict,
        "",
        "## 限制",
        "",
        "仅有三个训练种子；5k 指标方差高于标准 50k 评估；盲评判断嵌套于评审者与训练种子，原始 trial 数不能当作独立样本做夸大的显著性声明。结论只适用于当前实现、16 kimg 预算、NFE=1/2 与冻结采样协议。",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Final conclusion: {args.output}")


if __name__ == "__main__":
    main()
