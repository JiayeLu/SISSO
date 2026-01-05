#!/usr/bin/env python3
### get_coeff.py
# 这是一个用于导出 SISSO 模型系数并输出训练集预测及贡献项的脚本。
# 它读取训练数据和模型文件，计算系数，并将结果写入 CSV 文件。
# 该脚本假设存在特定的目录结构和文件格式。
# 使用时请确保已安装所需的 mt_sisso_scatter 模块。
###
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import math
import mt_sisso_scatter as m
import postprocess_utils as u


def write_coeff_csv(path: Path, tasks: List[str], coeffs: List[List[float]]) -> None:
    if not coeffs:
        raise ValueError("系数列表为空")
    desc_dim = len(coeffs[0]) - 1
    cols = ["task", "c0"] + [f"c{i}" for i in range(1, desc_dim + 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for task, coef in zip(tasks, coeffs):
            writer.writerow([task] + coef)


def write_train_preds_csv(
    path: Path,
    train_rows: List[Dict[str, float]],
    task_index: Dict[str, int],
    prepared_exprs: List[Tuple[str, Dict[str, str]]],
    coeffs: List[List[float]],
) -> None:
    desc_dim = len(prepared_exprs)
    cols = ["materials", "task", "true_T1", "pred_T1", "bias"] + [
        f"contrib_{i}" for i in range(1, desc_dim + 1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in train_rows:
            desc_values: List[float] = []
            for prepared, reverse_mapping in prepared_exprs:
                desc_values.append(m._evaluate_descriptor(prepared, reverse_mapping, row))
            materials = row["materials"]
            task = m._task_from_material(materials)
            idx = task_index.get(task)
            if idx is None:
                raise KeyError(f"Task {task} not found in training data")
            coef = coeffs[idx]
            bias = coef[0]
            contribs = [w * d for w, d in zip(coef[1:], desc_values)]
            pred = bias + sum(contribs)
            writer.writerow(
                [materials, task, row["T1"], pred, bias] + contribs
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="导出系数并输出 train.dat 预测及贡献项。")
    parser.add_argument("--model-rank", type=int, default=1, help="要使用的模型 rank（行号）")
    parser.add_argument(
        "--coeff-csv", type=Path, default=Path("coeffs.csv"), help="输出系数的 CSV 路径"
    )
    parser.add_argument(
        "--pred-csv",
        type=Path,
        default=Path("train_predictions.csv"),
        help="输出 train 预测和贡献项的 CSV 路径",
    )
    args = parser.parse_args()
    cfg = u.RunConfig(base=Path.cwd(), model_rank=args.model_rank)
    outputs = u.compute_outputs(cfg)
    task_index = outputs["task_index"] or u.build_task_index([])  # single task -> dummy
    coeffs_df = outputs["coeffs"]
    coeffs = coeffs_df[[c for c in coeffs_df.columns if c.startswith("c")]].values.tolist()
    tasks = coeffs_df["task"].tolist()

    # train predictions are already computed
    train_rows_df = outputs.get("train_preds")
    if train_rows_df is None:
        raise FileNotFoundError("train.dat not found")

    # rebuild task_index for multi-task; in single-task use single entry
    if outputs["single_job"]:
        task_index = {"all": 0}
    else:
        task_index = u.build_task_index(train_rows_df["materials"].tolist())
    # need prepared expressions again for contrib columns
    headers, train_rows, _, _, _, _, _, _ = u.load_run_artifacts(cfg)
    uspace = u.load_uspace(cfg.uspace_path)
    feature_ids = outputs["feature_ids"]
    prepared_exprs = []
    for fid in feature_ids:
        idx = fid - 1
        expr = uspace[idx]
        prepared_exprs.append(u.prepare_expression(expr, headers[1:]))

    write_coeff_csv(args.coeff_csv, tasks, coeffs)
    write_train_preds_csv(args.pred_csv, train_rows, task_index, prepared_exprs, coeffs)
    print(f"已写出系数: {args.coeff_csv}")
    print(f"已写出训练集预测: {args.pred_csv}")


if __name__ == "__main__":
    main()
