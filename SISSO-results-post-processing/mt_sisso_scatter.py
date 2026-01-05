#!/usr/bin/env python3
"""Generate train/predict scatter plots for a chosen MT-SISSO model rank.

Assumes the script is executed from the run directory that contains:
train.dat, predict.dat, SIS_subspaces/Uspace.expressions, SISSO.out, Models/.
The script auto-detects top*_D* and top*_D*_coeff files, evaluates the chosen
model rank, and saves a PNG with actual vs predicted scatter plots for both
train and predict sets.

为选定的 MT-SISSO 模型等级生成训练/预测散点图。

假设脚本在包含以下文件的运行目录下执行：
train.dat、predict.dat、SIS_subspaces/Uspace.expressions、SISSO.out 和 Models/ 目录。
脚本会自动检测 top*_D* 和 top*_D*_coeff 文件，评估选定的模型等级，并保存一个 PNG 文件，其中包含训练集和预测集的实际值与预测值散点图。
"""


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

import postprocess_utils as u


# 保留旧接口，转向新实现
def compute_model_outputs(model_rank: int = 1, base: Path | None = None):
    cfg = u.RunConfig(base=base or Path.cwd(), model_rank=model_rank)
    return u.compute_outputs(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MT-SISSO train/predict scatter for a model rank.")
    parser.add_argument("--model-rank", type=int, default=1, help="Rank (row) of the model to use")
    parser.add_argument(
        "--task-metrics-csv",
        type=Path,
        default=Path("task_metrics.csv"),
        help="当 ntask != 1 时，输出每个任务的 RMSE/MAE/R2/MAPE/MSE/SMAPE",
    )
    parser.add_argument("--output", type=Path, default=Path("scatter.png"), help="Output PNG file")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("predictions.csv"),
        help="CSV file to store true/pred values for train and predict",
    )
    args = parser.parse_args()

    cfg = u.RunConfig(
        base=Path.cwd(),
        model_rank=args.model_rank,
        scatter_png=args.output,
        preds_csv=args.csv_output,
        task_metrics_csv=args.task_metrics_csv,
    )
    outputs = u.compute_outputs(cfg)
    
    desc_dim = outputs["desc_dim"]
    single_job: bool = outputs["single_job"]

    datasets: List[tuple[str, List[Dict[str, Any]]]] = []
    if "train_preds" in outputs:
        datasets.append(("Train", outputs["train_preds"].to_dict(orient="records")))
    if "predict_preds" in outputs:
        datasets.append(("Predict", outputs["predict_preds"].to_dict(orient="records")))

    # 多任务时输出每个任务的指标
    if (not single_job) and datasets:
        metrics_rows = []
        for name, preds in datasets:
            df_metrics = u.per_task_metrics(preds)
            df_metrics["dataset"] = name.lower()
            metrics_rows.append(df_metrics)
        if metrics_rows:
            metrics_df = metrics_rows[0] if len(metrics_rows) == 1 else pd.concat(metrics_rows, ignore_index=True)
            metrics_df = metrics_df[
                ["dataset", "task", "RMSE", "MAE", "R2", "MAPE", "MSE", "SMAPE"]
            ]
            args.task_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(args.task_metrics_csv, index=False)
            print("Per-task metrics CSV:", args.task_metrics_csv)

    # 写预测 CSV
    u.write_predictions_csv(args.csv_output, datasets, desc_dim)

    # 绘图
    nplots = max(len(datasets), 1)
    fig, axes = plt.subplots(1, nplots, figsize=(6 * nplots, 5), sharex=True, sharey=True)
    if nplots == 1:
        axes = [axes]

    all_vals = [r["pred_T1"] for _, preds in datasets for r in preds]
    if not all_vals:
        raise ValueError("No data to plot")
    min_val, max_val = min(all_vals), max(all_vals)
    pad = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
    plot_min, plot_max = min_val - pad, max_val + pad

    for ax, (name, data) in zip(axes, datasets):
        for record in data:
            ax.scatter(record["true_T1"], record["pred_T1"], s=15, color="tab:blue", alpha=0.7)
        rmse, r2 = u.metrics_rmse_r2([(r["true_T1"], r["pred_T1"]) for r in data])
        ax.plot([plot_min, plot_max], [plot_min, plot_max], "k--", linewidth=1)
        ax.set_title(f"{name} (RMSE={rmse:.3f}, R2={r2:.3f})")
        ax.set_xlabel("True T1")
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
    axes[0].set_ylabel("Predicted T1")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print("Model file:", outputs['model_path'].name)
    print("Coeff file:", outputs['coeff_path'].name)
    print("Features:", outputs['feature_ids'])
    print("Saved scatter plot to:", args.output)


if __name__ == "__main__":
    main()
