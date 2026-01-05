"""Compute error metrics for every single-task SISSO model in the current run."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import List
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import postprocess_utils as u


def compute_metrics_for_models(base: Path, model_limit: int | None = None) -> pd.DataFrame:
    cfg = u.RunConfig(base=base)
    model_path = u.find_first(cfg.models_dir, r"top\\d+_D\\d+")
    ranks = []
    with model_path.open() as f:
        next(f, None)
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rank, _ = u.parse_model_line(ln)
            except ValueError:
                continue
            ranks.append(rank)
    ranks = sorted(set(ranks))
    if model_limit is not None:
        ranks = ranks[:model_limit]

    rows = []
    for rank in ranks:
        outputs = u.compute_outputs(u.RunConfig(base=base, model_rank=rank))
        coeff_df = outputs.get("coeffs")
        train_df = outputs.get("train_preds")
        predict_df = outputs.get("predict_preds")
        feature_ids = outputs.get("feature_ids", [])
        feature_exprs = outputs.get("feature_exprs", [])
        feature_ids_str = ";".join(str(fid) for fid in feature_ids)
        feature_exprs_str = " ; ".join(feature_exprs)

        for dataset_name, df in (("train", train_df), ("predict", predict_df)):
            if df is None:
                continue
            pairs = list(zip(df["true_T1"], df["pred_T1"]))
            metrics = u.regression_metrics(pairs)
            rows.append(
                {
                    "Model_Rank": rank,
                    "Dataset": dataset_name,
                    "Feature_IDs": feature_ids_str,
                    "Feature_Exprs": feature_exprs_str,
                    **metrics,
                    "coeff": coeff_df.to_dict(orient="records") if coeff_df is not None else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute metrics for each single-task SISSO model.")
    parser.add_argument(
        "--sisso-dir",
        type=Path,
        default=Path.cwd(),
        help="Path to SISSO run directory (defaults to current working directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_metrics.csv"),
        help="CSV file to write metrics.",
    )
    parser.add_argument(
        "--model-limit",
        type=int,
        default=None,
        help="只处理前 N 个模型（按 rank 排序）；默认全部",
    )
    args = parser.parse_args()
    df = compute_metrics_for_models(args.sisso_dir, model_limit=args.model_limit).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved metrics for {len(df)} entries to {args.output}")


if __name__ == "__main__":
    main()
