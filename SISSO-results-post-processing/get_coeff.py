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

import mt_sisso_scatter as m


def _prepare_expressions(uspace: Sequence[str], feature_ids: List[int], feature_cols: Sequence[str]):
    prepared = []
    for fid in feature_ids:
        idx = fid - 1
        if idx < 0 or idx >= len(uspace):
            raise ValueError(f"Feature ID {fid} out of range for Uspace of size {len(uspace)}")
        expr = uspace[idx]
        prepared.append(m._prepare_expression(expr, feature_cols))
    return prepared


def _task_order(task_index: Dict[str, int]) -> List[str]:
    return [t for t, _ in sorted(task_index.items(), key=lambda kv: kv[1])]


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

    base = Path.cwd()
    train_path = base / "train.dat"
    uspace_path = base / "SIS_subspaces" / "Uspace.expressions"
    models_dir = base / "Models"

    model_path = m._find_first(models_dir, r"top\d+_D\d+")
    coeff_path = m._find_first(models_dir, r"top\d+_D\d+_coeff")

    headers_train, train_rows = m._read_table(train_path)
    feature_cols = headers_train[1:]  # 排除 materials

    task_index = m._build_task_index(row["materials"] for row in train_rows)
    tasks = _task_order(task_index)

    uspace = m._load_uspace(uspace_path)
    feature_ids = m._load_model_features(model_path, args.model_rank)
    desc_dim = len(feature_ids)
    coeffs = m._load_coefficients(coeff_path, args.model_rank, len(task_index), desc_dim)

    prepared_exprs = _prepare_expressions(uspace, feature_ids, feature_cols)
    print(prepared_exprs)
    write_coeff_csv(args.coeff_csv, tasks, coeffs)
    write_train_preds_csv(args.pred_csv, train_rows, task_index, prepared_exprs, coeffs)
    print(f"已写出系数: {args.coeff_csv}")
    print(f"已写出训练集预测: {args.pred_csv}")


if __name__ == "__main__":
    main()
