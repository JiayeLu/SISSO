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
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import csv

import matplotlib.pyplot as plt

# 从材料名提取任务标记 / Extract task tag from material name
def _task_from_material(material: str) -> str:
    parts = material.split("_")
    return parts[-1] if parts else material

# 根据材料顺序构建任务索引 / Build task index based on material order
def _build_task_index(materials: Iterable[str]) -> Dict[str, int]:
    order: Dict[str, int] = {}
    for m in materials:
        t = _task_from_material(m)
        if t not in order:
            order[t] = len(order)
    return order

# 读取数据表 / Read data table
def _read_table(path: Path) -> Tuple[List[str], List[Dict[str, float]]]:
    with path.open() as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"No data in {path}")
    headers = lines[0].split()
    data: List[Dict[str, float]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != len(headers):
            raise ValueError(f"Line has {len(parts)} columns but expected {len(headers)}: {line}")
        row: Dict[str, float] = {headers[0]: parts[0]}
        for h, val in zip(headers[1:], parts[1:]):
            row[h] = float(val)
        data.append(row)
    return headers, data

# 读取 Uspace 表达式 / Load expressions from Uspace
def _load_uspace(uspace_path: Path) -> List[str]:
    expressions: List[str] = []
    with uspace_path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            expr = stripped.split("SIS_score")[0].strip()
            expressions.append(expr)
    return expressions

# 将名称转为安全的 Python 变量名 / Make a safe Python identifier
def _safe_name(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if safe and safe[0].isdigit():
        safe = f"_{safe}"
    return safe

# 准备表达式用于 eval / Prepare expression for eval
def _prepare_expression(expr: str, columns: Sequence[str]) -> Tuple[str, Dict[str, str]]:
    mapping = {col: _safe_name(col) for col in columns}
    sorted_cols = sorted(columns, key=len, reverse=True)
    # SISSO expressions use ^ for exponentiation; Python expects **.
    prepared = expr.replace("^", "**")
    for col in sorted_cols:
        pattern = re.compile(rf"(?<![0-9A-Za-z_]){re.escape(col)}(?![0-9A-Za-z_])")
        prepared = pattern.sub(mapping[col], prepared)
    reverse_mapping = {v: k for k, v in mapping.items()}
    return prepared, reverse_mapping

# 计算描述符值 / Evaluate descriptor value
def _evaluate_descriptor(expr: str, reverse_mapping: Dict[str, str], row: Dict[str, float]) -> float:
    safe_values = {safe: row[orig] for safe, orig in reverse_mapping.items()}
    return float(eval(expr, {}, safe_values))

# 解析模型行得到排名与特征 ID / Parse model line for rank and feature ids
def _parse_model_line(line: str) -> Tuple[int, List[int]]:
    parts = line.strip().split()
    if not parts or not parts[0].isdigit():
        raise ValueError(f"Invalid model line: {line}")
    rank = int(parts[0])
    match = re.search(r"\(([^)]+)\)", line)
    if not match:
        raise ValueError(f"Cannot find feature IDs in line: {line}")
    ids = [int(x) for x in match.group(1).split()]
    return rank, ids

# 从模型文件读取特征 ID / Load feature IDs from model file
def _load_model_features(model_path: Path, model_rank: int) -> List[int]:
    with model_path.open() as f:
        next(f)  # header
        for line in f:
            if not line.strip():
                continue
            rank, ids = _parse_model_line(line)
            if rank == model_rank:
                return ids
    raise ValueError(f"Model rank {model_rank} not found in {model_path}")

# 读取指定模型的系数 / Load coefficients for a model rank
def _load_coefficients(coeff_path: Path, model_rank: int, ntasks: int, desc_dim: int) -> List[List[float]]:
    needed = (desc_dim + 1) * ntasks
    with coeff_path.open() as f:
        next(f)  # header
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if not parts:
                continue
            if int(parts[0]) != model_rank:
                continue
            coeffs = [float(x) for x in parts[1:]]
            if len(coeffs) != needed:
                raise ValueError(
                    f"Expected {needed} coefficients for model {model_rank}, found {len(coeffs)}"
                )
            grouped = [coeffs[i : i + desc_dim + 1] for i in range(0, len(coeffs), desc_dim + 1)]
            return grouped
    raise ValueError(f"Coefficients for model rank {model_rank} not found in {coeff_path}")

# 在目录中寻找匹配的第一个文件 / Find first file matching pattern
def _find_first(models_dir: Path, pattern: str) -> Path:
    regex = re.compile(pattern)
    for candidate in sorted(models_dir.iterdir()):
        if regex.fullmatch(candidate.name):
            return candidate
    raise FileNotFoundError(f"No file matching {pattern} in {models_dir}")

# 计算 RMSE 和 R2 / Compute RMSE and R2 metrics

def _metrics(pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not pairs:
        return float("nan"), float("nan")
    diffs = [(t - p) for t, p in pairs]
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    if len(pairs) <= 1:
        return rmse, float("nan")
    true_vals = [t for t, _ in pairs]
    mean_true = sum(true_vals) / len(true_vals)
    ss_res = sum((t - p) ** 2 for t, p in pairs)
    ss_tot = sum((t - mean_true) ** 2 for t in true_vals)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return rmse, r2

# 对数据集进行预测 / Predict dataset values
def _predict_dataset(
    rows: List[Dict[str, float]],
    feature_ids: List[int],
    prepared_exprs: List[Tuple[str, Dict[str, str]]],
    coeffs: List[List[float]],
    task_index: Dict[str, int],
) -> List[Tuple[float, float, str, str]]:
    results: List[Tuple[float, float, str, str]] = []
    for row in rows:
        desc_values: List[float] = []
        for prepared, reverse_mapping in prepared_exprs:
            desc_values.append(_evaluate_descriptor(prepared, reverse_mapping, row))
        materials = row["materials"]
        task = _task_from_material(materials)
        idx = task_index.get(task)
        if idx is None:
            raise KeyError(f"Task {task} not found in training data")
        coef = coeffs[idx]
        pred = coef[0] + sum(w * d for w, d in zip(coef[1:], desc_values))
        results.append((row["T1"], pred, task, materials))
    return results

# 主流程：读取、预测并绘图 / Main pipeline: load, predict, and plot
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MT-SISSO train/predict scatter for a model rank.")
    parser.add_argument("--model-rank", type=int, default=1, help="Rank (row) of the model to use")
    # parser.add_argument(
    #     "--output", type=Path, default=Path("scatter.png"), help="Output PNG file"
    # )
    # parser.add_argument(
    #     "--csv-output",
    #     type=Path,
    #     default=Path("predictions.csv"),
    #     help="CSV file to store true/pred values for train and predict",
    # )
    args = parser.parse_args()
    
    base = Path.cwd()
    
    train_path = base / "train.dat"
    predict_path = base / "predict.dat"
    uspace_path = base / "SIS_subspaces" / "Uspace.expressions"
    models_dir = base / "Models"
    csv_path = base / "predictions.csv"
    output_fig = base / "scatter.png"

    model_path = _find_first(models_dir, r"top\d+_D\d+")
    coeff_path = _find_first(models_dir, r"top\d+_D\d+_coeff")

    has_train = train_path.exists()
    has_predict = predict_path.exists()
    if not (has_train or has_predict):
        raise FileNotFoundError("缺少 train.dat 或 predict.dat，至少需要一个文件")

    headers_train: List[str] = []
    train_rows: List[Dict[str, float]] = []
    if has_train:
        headers_train, train_rows = _read_table(train_path)

    headers_pred: List[str] = []
    predict_rows: List[Dict[str, float]] = []
    if has_predict:
        headers_pred, predict_rows = _read_table(predict_path)

    if has_train and has_predict and headers_train != headers_pred:
        raise ValueError("train.dat 和 predict.dat 的表头不一致")

    headers = headers_train or headers_pred
    if not headers:
        raise ValueError("无法获取表头信息")

    task_source = train_rows if train_rows else predict_rows
    task_index = _build_task_index(row["materials"] for row in task_source)

    uspace = _load_uspace(uspace_path)
    feature_ids = _load_model_features(model_path, args.model_rank)
    desc_dim = len(feature_ids)
    coeffs = _load_coefficients(coeff_path, args.model_rank, len(task_index), desc_dim)

    prepared_expressions: List[Tuple[str, Dict[str, str]]] = []
    for fid in feature_ids:
        idx = fid - 1
        if idx < 0 or idx >= len(uspace):
            raise ValueError(f"Feature ID {fid} out of range for Uspace of size {len(uspace)}")
        expr = uspace[idx]
        prepared_expressions.append(_prepare_expression(expr, headers[1:]))  # exclude materials

    datasets = []
    if has_train:
        train_preds = _predict_dataset(train_rows, feature_ids, prepared_expressions, coeffs, task_index)
        datasets.append(("Train", train_preds))
    if has_predict:
        predict_preds = _predict_dataset(predict_rows, feature_ids, prepared_expressions, coeffs, task_index)
        datasets.append(("Predict", predict_preds))

    # Write combined CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "materials", "task", "true_T1", "pred_T1"])
        for name, preds in datasets:
            for true, pred, task, materials in preds:
                writer.writerow([name.lower(), materials, task, f"{true:.6f}", f"{pred:.6f}"])

    nplots = max(len(datasets), 1)
    fig, axes = plt.subplots(1, nplots, figsize=(6 * nplots, 5), sharex=True, sharey=True)
    if nplots == 1:
        axes = [axes]

    all_vals = [p for _, preds in datasets for _, p, _, _ in preds]
    if not all_vals:
        raise ValueError("No data to plot")
    min_val, max_val = min(all_vals), max(all_vals)
    pad = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
    plot_min, plot_max = min_val - pad, max_val + pad

    for ax, (name, data) in zip(axes, datasets):
        for true, pred, _, _ in data:
            ax.scatter(true, pred, s=15, color="tab:blue", alpha=0.7)
        rmse, r2 = _metrics([(t, p) for t, p, _, _ in data])
        ax.plot([plot_min, plot_max], [plot_min, plot_max], "k--", linewidth=1)
        ax.set_title(f"{name} (RMSE={rmse:.3f}, R2={r2:.3f})")
        ax.set_xlabel("True T1")
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
    axes[0].set_ylabel("Predicted T1")
    fig.tight_layout()
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, dpi=200)
    print("Model file:", model_path.name)
    print("Coeff file:", coeff_path.name)
    print("Features:", feature_ids)
    print("Saved scatter plot to:", output_fig)


if __name__ == "__main__":
    main()
