from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

# --------------------------------------------------------------------------- #
# Data model / configuration
# --------------------------------------------------------------------------- #


@dataclass
class RunConfig:
    base: Path
    model_rank: int = 1
    scatter_png: Path | None = None
    preds_csv: Path | None = None
    task_metrics_csv: Path | None = None

    @property
    def train_path(self) -> Path:
        return self.base / "train.dat"

    @property
    def predict_path(self) -> Path:
        return self.base / "predict.dat"

    @property
    def uspace_path(self) -> Path:
        return self.base / "SIS_subspaces" / "Uspace.expressions"

    @property
    def models_dir(self) -> Path:
        return self.base / "Models"

    @property
    def sisso_in(self) -> Path:
        return self.base / "SISSO.in"


# --------------------------------------------------------------------------- #
# Parsing utilities
# --------------------------------------------------------------------------- #

_FUNC_REPLACEMENTS = {
    "sqrt": "math.sqrt",
    "exp": "math.exp",
    "log": "math.log",
    "ln": "math.log",
    "sin": "math.sin",
    "cos": "math.cos",
    "tan": "math.tan",
    "asin": "math.asin",
    "acos": "math.acos",
    "atan": "math.atan",
    "sinh": "math.sinh",
    "cosh": "math.cosh",
    "tanh": "math.tanh",
    "abs": "abs",
    "cbrt": "math.cbrt",
}


def task_from_material(material: str) -> str:
    parts = material.split("_")
    return parts[-1] if parts else material


def build_task_index(materials: Iterable[str]) -> Dict[str, int]:
    """Assign a stable index for each task (only use in multi-task)."""
    order: Dict[str, int] = {}
    for m in materials:
        t = task_from_material(m)
        if t not in order:
            order[t] = len(order)
    return order


def read_table(path: Path) -> Tuple[List[str], List[Dict[str, float]]]:
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


def parse_int_list(raw: str) -> List[int]:
    cleaned = raw.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    cleaned = cleaned.replace(" ", "")
    if not cleaned:
        return []
    return [int(part) for part in cleaned.split(",") if part]


def read_sisso_config(sisso_path: Path) -> Tuple[int, List[int]]:
    if not sisso_path.exists():
        raise FileNotFoundError(f"SISSO.in not found at {sisso_path}")
    ntasks_declared: int | None = None
    nsample: List[int] = []
    with sisso_path.open() as f:
        for line in f:
            content = line.split("!")[0].strip()
            if not content:
                continue
            ntask_match = re.search(r"ntasks?\s*=\s*([^\s]+)", content)
            if ntask_match:
                try:
                    ntasks_declared = int(ntask_match.group(1))
                except ValueError:
                    pass
            nsample_match = re.search(r"nsample\s*=\s*([^\s]+)", content)
            if nsample_match:
                try:
                    nsample = parse_int_list(nsample_match.group(1))
                except ValueError:
                    nsample = []
    if ntasks_declared is None:
        ntasks_declared = len(nsample) if nsample else 1
    return ntasks_declared, nsample


def load_uspace(uspace_path: Path) -> List[str]:
    expressions: List[str] = []
    with uspace_path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            expr = stripped.split("SIS_score")[0].strip()
            expressions.append(expr)
    return expressions


def safe_name(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if safe and safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def prepare_expression(expr: str, columns: Sequence[str]) -> Tuple[str, Dict[str, str]]:
    mapping = {col: safe_name(col) for col in columns}
    sorted_cols = sorted(columns, key=len, reverse=True)
    prepared = expr.replace("^", "**")
    for src, dst in _FUNC_REPLACEMENTS.items():
        prepared = re.sub(rf"(?<![0-9A-Za-z_]){src}(?![0-9A-Za-z_])", dst, prepared)
    for col in sorted_cols:
        pattern = re.compile(rf"(?<![0-9A-Za-z_]){re.escape(col)}(?![0-9A-Za-z_])")
        prepared = pattern.sub(mapping[col], prepared)
    reverse_mapping = {v: k for k, v in mapping.items()}
    return prepared, reverse_mapping


def evaluate_descriptor(expr: str, reverse_mapping: Dict[str, str], row: Dict[str, float]) -> float:
    safe_values = {safe: row[orig] for safe, orig in reverse_mapping.items()}
    return float(eval(expr, {"math": math}, safe_values))


def parse_model_line(line: str) -> Tuple[int, List[int]]:
    parts = line.strip().split()
    if not parts or not parts[0].isdigit():
        raise ValueError(f"Invalid model line: {line}")
    rank = int(parts[0])
    match = re.search(r"\(([^)]+)\)", line)
    if not match:
        raise ValueError(f"Cannot find feature IDs in line: {line}")
    ids = [int(x) for x in match.group(1).split()]
    return rank, ids


def load_model_features(model_path: Path, model_rank: int) -> List[int]:
    with model_path.open() as f:
        next(f, None)
        for line in f:
            if not line.strip():
                continue
            rank, ids = parse_model_line(line)
            if rank == model_rank:
                return ids
    raise ValueError(f"Model rank {model_rank} not found in {model_path}")


def load_coefficients(
    coeff_path: Path, model_rank: int, ntasks: int, desc_dim: int, single_job: bool
) -> List[List[float]]:
    needed = (desc_dim + 1) * ntasks

    with coeff_path.open() as f:
        next(f, None)
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if not parts or int(parts[0]) != model_rank:
                continue
            coeffs = [float(x) for x in parts[1:]]
            if (not single_job) and len(coeffs) != needed:
                raise ValueError(
                    f"Expected {needed} coefficients for model {model_rank}, found {len(coeffs)}"
                )
            grouped = [coeffs[i : i + desc_dim + 1] for i in range(0, len(coeffs), desc_dim + 1)]
            return grouped
    raise ValueError(f"Coefficients for model rank {model_rank} not found in {coeff_path}")


def find_first(models_dir: Path, pattern: str) -> Path:
    regex = re.compile(pattern)
    for candidate in sorted(models_dir.iterdir()):
        if regex.fullmatch(candidate.name):
            return candidate
    raise FileNotFoundError(f"No file matching {pattern} in {models_dir}")


# --------------------------------------------------------------------------- #
# Metrics / prediction helpers
# --------------------------------------------------------------------------- #


def metrics_rmse_r2(pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
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


def _safe_mape(pairs: Iterable[Tuple[float, float]]) -> float:
    values: List[float] = []
    for true, pred in pairs:
        if true == 0:
            continue
        values.append(abs((true - pred) / true))
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _safe_smape(pairs: Iterable[Tuple[float, float]]) -> float:
    values = []
    for true, pred in pairs:
        denom = abs(true) + abs(pred)
        if denom == 0:
            continue
        values.append(2 * abs(pred - true) / denom)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def regression_metrics(pairs: List[Tuple[float, float]]) -> Dict[str, float]:
    if not pairs:
        return {k: float("nan") for k in ["RMSE", "MSE", "MAE", "MAPE", "SMAPE", "R2"]}
    diffs = [t - p for t, p in pairs]
    mse = sum(d * d for d in diffs) / len(diffs)
    rmse = math.sqrt(mse)
    mae = sum(abs(d) for d in diffs) / len(diffs)
    mape = _safe_mape(pairs)
    smape = _safe_smape(pairs)
    _, r2 = metrics_rmse_r2(pairs)
    return {"RMSE": round(rmse,2), "MSE": round(mse,2), "MAE": round(mae,2), "MAPE": round(mape,2), "SMAPE": round(smape,2), "R2": round(r2,2)}


def predict_dataset(
    rows: List[Dict[str, float]],
    feature_ids: List[int],
    prepared_exprs: List[Tuple[str, Dict[str, str]]],
    coeffs: List[List[float]],
    task_index: Dict[str, int] | None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for row in rows:
        target_key = list(row.keys())[1]
        desc_values: List[float] = []
        for prepared, reverse_mapping in prepared_exprs:
            desc_values.append(evaluate_descriptor(prepared, reverse_mapping, row))
        materials = row["materials"]
        task = task_from_material(materials)
        idx = 0
        if len(coeffs) > 1:
            if task_index is None:
                raise KeyError("task_index missing for multi-task prediction")
            idx = task_index.get(task)
            if idx is None:
                raise KeyError(f"Task {task} not found in training data")
        coef = coeffs[idx]
        contribs = [w * d for w, d in zip(coef[1:], desc_values)]
        pred = coef[0] + sum(contribs)
        results.append(
            {
                "true_T1": row[target_key],
                "pred_T1": pred,
                "task": task,
                "materials": materials,
                "desc_values": desc_values,
                "desc_contribs": contribs,
            }
        )
    return results


# --------------------------------------------------------------------------- #
# High-level loaders
# --------------------------------------------------------------------------- #


def load_run_artifacts(cfg: RunConfig) -> Tuple[
    List[str], List[Dict[str, float]], List[str], List[Dict[str, float]], Path, Path, int, bool
]:
    has_train = cfg.train_path.exists()
    has_predict = cfg.predict_path.exists()
    if not (has_train or has_predict):
        raise FileNotFoundError("缺少 train.dat 或 predict.dat，至少需要一个文件")

    headers_train: List[str] = []
    train_rows: List[Dict[str, float]] = []
    if has_train:
        headers_train, train_rows = read_table(cfg.train_path)

    headers_pred: List[str] = []
    predict_rows: List[Dict[str, float]] = []
    if has_predict:
        headers_pred, predict_rows = read_table(cfg.predict_path)

    if has_train and has_predict and headers_train != headers_pred:
        raise ValueError("train.dat 和 predict.dat 的表头不一致")

    headers = headers_train or headers_pred
    if not headers:
        raise ValueError("无法获取表头信息")

    model_path = find_first(cfg.models_dir, r"top\d+_D\d+")
    coeff_path = find_first(cfg.models_dir, r"top\d+_D\d+_coeff")
    ntasks_config, _ = read_sisso_config(cfg.sisso_in)
    single_job = ntasks_config == 1

    return headers, train_rows, headers_pred, predict_rows, model_path, coeff_path, ntasks_config, single_job


def compute_outputs(cfg: RunConfig) -> Dict[str, Any]:
    headers, train_rows, _, predict_rows, model_path, coeff_path, ntasks_config, single_job = load_run_artifacts(cfg)
    task_source = train_rows if train_rows else predict_rows
    task_index = None if single_job else build_task_index(row["materials"] for row in task_source)
    # print(task_index)
    uspace = load_uspace(cfg.uspace_path)
    feature_ids = load_model_features(model_path, cfg.model_rank)
    desc_dim = len(feature_ids)
    coeffs = load_coefficients(coeff_path, cfg.model_rank, ntasks_config, desc_dim, single_job)

    prepared_expressions: List[Tuple[str, Dict[str, str]]] = []
    for fid in feature_ids:
        idx = fid - 1
        if idx < 0 or idx >= len(uspace):
            raise ValueError(f"Feature ID {fid} out of range for Uspace of size {len(uspace)}")
        expr = uspace[idx]
        prepared_expressions.append(prepare_expression(expr, headers[1:]))  # exclude materials

    outputs: Dict[str, Any] = {
        "feature_ids": feature_ids,
        "feature_exprs": [uspace[fid - 1] for fid in feature_ids],
        "coeffs": pd.DataFrame(
            [
                {
                    "task": (task if task_index is not None else "all"),
                    **{f"c{j}": c for j, c in enumerate(coef, start=0)},
                }
                for task, coef in zip(
                    (task_index and {v: k for k, v in task_index.items()}.values()) or ["all"], coeffs
                )
            ]
        ),
    }

    def preds_to_df(preds: List[Dict[str, Any]]) -> pd.DataFrame:
        data = []
        for record in preds:
            row: Dict[str, Any] = {
                "materials": record["materials"],
                "task": record["task"],
                "true_T1": record["true_T1"],
                "pred_T1": record["pred_T1"],
            }
            for i, val in enumerate(record["desc_values"]):
                row[f"desc{i+1}_value"] = val
            for i, contrib in enumerate(record["desc_contribs"]):
                row[f"c{i+1} * desc"] = contrib
            data.append(row)
        return pd.DataFrame(data)

    prepared = prepared_expressions
    if train_rows:
        train_preds = predict_dataset(train_rows, feature_ids, prepared, coeffs, task_index)
        outputs["train_preds"] = preds_to_df(train_preds)
    if predict_rows:
        predict_preds = predict_dataset(predict_rows, feature_ids, prepared, coeffs, task_index)
        outputs["predict_preds"] = preds_to_df(predict_preds)
    outputs["task_index"] = task_index
    outputs["desc_dim"] = desc_dim
    outputs["single_job"] = single_job
    outputs["model_path"] = model_path
    outputs["coeff_path"] = coeff_path
    return outputs


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def write_predictions_csv(path: Path, datasets: List[Tuple[str, List[Dict[str, Any]]]], desc_dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    desc_value_headers = [f"desc{i+1}_value" for i in range(desc_dim)]
    contrib_headers = [f"c{i+1} * desc" for i in range(desc_dim)]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset", "materials", "task", "true_T1", "pred_T1", *desc_value_headers, *contrib_headers]
        )
        for name, preds in datasets:
            for record in preds:
                # record 可能来自 raw 预测（含 desc_values/desc_contribs 列表），也可能来自 DataFrame to_dict 展开后的列
                has_list = "desc_values" in record and "desc_contribs" in record
                has_expanded = desc_value_headers and desc_value_headers[0] in record
                if has_list:
                    desc_vals = record["desc_values"]
                    contrib_vals = record["c * desc"]
                else:  # 已展开列（desc{i}_value / C{i}_times_desc）
                    desc_vals = [record.get(h, float("nan")) for h in desc_value_headers]
                    contrib_vals = [record.get(h, float("nan")) for h in contrib_headers]

                writer.writerow(
                    [
                        name.lower(),
                        record["materials"],
                        record["task"],
                        f"{record['true_T1']:.6f}",
                        f"{record['pred_T1']:.6f}",
                        *[f"{v:.6f}" for v in desc_vals],
                        *[f"{c:.6f}" for c in contrib_vals],
                    ]
                )


def per_task_metrics(preds: List[Dict[str, Any]]) -> pd.DataFrame:
    by_task: Dict[str, List[Tuple[float, float]]] = {}
    for record in preds:
        by_task.setdefault(record["task"], []).append((record["true_T1"], record["pred_T1"]))
    metrics_rows: List[Dict[str, Any]] = []
    for task, pairs in sorted(by_task.items()):
        metrics = regression_metrics(pairs)
        metrics_rows.append({"task": task, **metrics})
    return pd.DataFrame(metrics_rows)
