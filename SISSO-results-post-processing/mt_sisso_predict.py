#!/usr/bin/env python3
"""Predict MT-SISSO outputs for new data and report per-task metrics.

Inputs are fixed to the iter013 run to simplify usage. The script automatically
finds the latest top*_D* model file and its matching coefficient file via
regular expressions, reads SISSO.out to obtain the training RMSE, evaluates the
selected model rank on predict.dat, and writes per-task RMSE/R2 along with the
training RMSE.
"""

from __future__ import annotations
import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCATTER_DIR = Path(__file__).parent / "SISSO-results-post-processing"
if SCATTER_DIR.is_dir():
    sys.path.insert(0, str(SCATTER_DIR))

from mt_sisso_scatter import (  # type: ignore
    _build_task_index,
    _find_first,
    _load_coefficients,
    _load_model_features,
    _load_uspace,
    _metrics,
    _prepare_expression,
    _read_table,
    _task_from_material,
)

# Math functions allowed inside descriptor expressions
ALLOWED_FUNCS = {
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "tanh": math.tanh,
    "pow": pow,
}


def _evaluate_descriptor(expr: str, reverse_mapping: Dict[str, str], row: Dict[str, float]) -> float:
    safe_values = {safe: row[orig] for safe, orig in reverse_mapping.items()}
    env = {**ALLOWED_FUNCS, **safe_values}
    return float(eval(expr, {"__builtins__": {}}, env))


def _parse_train_rmse(sisso_out: Path) -> list[float]:
    train_rmse: List[float] = []
    pattern = re.compile(r"RMSE and MaxAE:\s*([0-9.E+-]+)")
    for line in (sisso_out.read_text().splitlines()):
        match = pattern.search(line)
        if match:
            train_rmse.append(float(match.group(1)))
    return train_rmse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MT-SISSO predictions per task.")
    parser.add_argument("--model-rank", type=int, default=1, help="Rank (row) of the model to use")
    parser.add_argument(
        "--output", type=Path, default=Path("predict_metrics.txt"), help="Output file"
    )
    args = parser.parse_args()



    base = Path.cwd()
    train_path = base / "train.dat"
    predict_path = base / "predict.dat"
    uspace_path = base / "SIS_subspaces" / "Uspace.expressions"
    sisso_out_path = base / "SISSO.out"
    models_dir = base / "Models"


    model_path = _find_first(models_dir, r"top\d+_D\d+")
    coeff_path = _find_first(models_dir, r"top\d+_D\d+_coeff")
    train_rmse = _parse_train_rmse(sisso_out_path)

    headers_train, train_rows = _read_table(train_path)
    headers_pred, predict_rows = _read_table(predict_path)
    if headers_train != headers_pred:
        raise ValueError("train.dat and predict.dat have different headers")

    task_index = _build_task_index(row["materials"] for row in train_rows)
    predict_tasks = [_task_from_material(row["materials"]) for row in predict_rows]

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
        prepared_expressions.append(_prepare_expression(expr, headers_pred[1:]))  # exclude materials
    task_pairs: Dict[str, List[Tuple[float, float]]] = {t: [] for t in task_index}
    for row, task in zip(predict_rows, predict_tasks):
        desc_values: List[float] = []
        for prepared, reverse_mapping in prepared_expressions:
            desc_values.append(_evaluate_descriptor(prepared, reverse_mapping, row))
        idx = task_index.get(task)
        if idx is None:
            raise KeyError(f"Task {task} not found in training data")
        coef = coeffs[idx]
        pred = coef[0] + sum(w * d for w, d in zip(coef[1:], desc_values))
        task_pairs[task].append((row["T1"], pred))

    report_lines = ["task_index task_name train_RMSE predict_RMSE R2 "]
    all_pairs: List[Tuple[float, float]] = []
    for task, idx in sorted(task_index.items(), key=lambda x: x[1]):
        rmse, r2 = _metrics(task_pairs[task])
        report_lines.append(f"{idx+1} {task} {train_rmse[idx]} {rmse:.6f} {r2:.6f}")
        all_pairs.extend(task_pairs[task])
    overall_rmse, _ = _metrics(all_pairs)
    args.output.write_text(
        "\n".join(report_lines) + "\n" + f"Overall predict_RMSE: {overall_rmse:.6f}"
    )

    print("Model file:", model_path.name)
    print("Coeff file:", coeff_path.name)
    print("Model rank:", args.model_rank)
    print("Features:", feature_ids)
    # print("Train RMSE:", f"{train_rmse:.6f}")
    print("Report written to:", args.output)


if __name__ == "__main__":
    main()
