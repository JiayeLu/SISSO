import pandas as pd
import glob, os, re, sys
from pathlib import Path

# Reuse parsing helpers from mt_sisso_scatter to avoid duplicate logic.
MT_SCATTER_DIR = Path(__file__).parent / "2D1-small-M-all"
if MT_SCATTER_DIR.is_dir():
    sys.path.insert(0, str(MT_SCATTER_DIR))

from mt_sisso_scatter import _load_uspace, _parse_model_line  # type: ignore

# 解析 top*_D* 模型文件 
# Parse top*_D* model summary file
def _parse_top_file(path):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("Rank"):
                continue
            try:
                rank, ids = _parse_model_line(ln)
            except ValueError:
                continue

            parts = ln.split()
            if len(parts) < 3:
                continue

            rows.append({
                "Rank": rank,
                "RMSE": float(parts[1]),
                "MaxAE": float(parts[2]),
                "Feature_ID": ids,
            })
    return pd.DataFrame(rows)

# 解析系数文件 
# Parse coefficient file
def _parse_coeff_file(path, n_dim):
    # 需要传入维度 D（特征数量），系数总数 = D+1
    coeff_cols = [f"c{i}" for i in range(n_dim + 1)]
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.lower().startswith("model"):
                continue
            nums = re.findall(r"[+-]?\d+\.\d+(?:[Ee][+-]?\d+)?|[+-]?\d+", ln)
            if len(nums) < (1 + n_dim + 1):  # model_id + (D+1) coefficients
                continue
            model_id = int(nums[0])
            coeffs = [float(x) for x in nums[1 : 1 + n_dim + 1]]
            rows.append({"Model_ID": model_id, **dict(zip(coeff_cols, coeffs))})
    return pd.DataFrame(rows)

# 读取 Uspace 表达式
# Load Uspace expressions
def _parse_uspace(path):
    exprs = _load_uspace(Path(path))
    rows = [{"Feature_ID": idx, "expr": expr} for idx, expr in enumerate(exprs, start=1)]
    return pd.DataFrame(rows).set_index("Feature_ID")

# 汇总特征表达式与系数 / Gather feature expressions and coefficients for a run
def get_feature_expressions(root_dir):
    # Read the top*_D00* file
    model_files = glob.glob(os.path.join(f'{root_dir}/Models/', "top*_D*"))
    model_files = [p for p in model_files if not p.endswith("_coeff")]
    top_rank = model_files[0]
    exprs_coeff = "".join([top_rank,'_coeff'])
    # Get Feature_IDs as list of integers (2D/3D...)
    df_fc = _parse_top_file(top_rank)
    if df_fc.empty:
        raise ValueError(f"{top_rank} 解析后空 DataFrame")
    # Read the top*_D00*_coeff file
    df_fi = _parse_coeff_file(exprs_coeff, n_dim=len(df_fc.iloc[0]["Feature_ID"]))
    #Read Expressions
    df_ep = _parse_uspace(f'{root_dir}/SIS_subspaces/Uspace.expressions')
    id_to_expr = df_ep["expr"].to_dict()

# Get Feature_IDs as list of integers (2D/3D...)
    # start_idx = df_fc.columns.get_loc("Feature_ID")
    # feat_cols = df_fc.columns[start_idx:]
    # feat_str = df_fc[feat_cols].astype(str).apply(lambda row: " ".join(row.values), axis=1)
    # df_fc["Feature_IDs"] = feat_str.str.findall(r"\d+").apply(lambda xs: [int(x) for x in xs])

# Read the top*_D00*_coeff file
    
    # df = pd.read_fwf(exprs_coeff, delim_whitespace=True)
    # df_fi = df.iloc[:,[1,2,4]]
    # df_fi.columns = ["c0", "c1", "c2"]

#Read Expressions
    # df_ep = pd.read_fwf(f'{root_dir}/SIS_subspaces/Uspace.expressions', delim_whitespace=True, header=None)
    # df_ep["expr"] = df_ep[0].astype(str).str.split().str[0]
    # df_ep.index += 1
    # id_to_expr = df_ep["expr"].to_dict()

    # inflect feature ids to expressions
    df_fc["Feature_exprs"] = df_fc["Feature_ID"].apply(
        lambda ids: [id_to_expr[i] for i in ids]
    )
    return df_fc, df_fi

# 收集全部任务结果 
# Collect all task results
def collect_all_runs(parent_dir):
    all_fc = []   # 收集每个任务的 df_fc
    all_fi = []   # 收集每个任务的 df_fi

    for entry in os.listdir(parent_dir):
        run_dir = os.path.join(parent_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        
        models_dir = os.path.join(run_dir, "Models")
        uspace_dir = os.path.join(run_dir, "SIS_subspaces")
        # 必须存在这两个才能视为 SISSO 任务目录
        if not os.path.isdir(models_dir) or not os.path.isdir(uspace_dir):
            continue

        print(f"[INFO] Processing task: {run_dir}")

        try:
            df_fc, df_fi = get_feature_expressions(run_dir)
        except Exception as e:
            print(f"[WARN] Failed to parse: {run_dir}")
            print("       Reason:", e)
            continue

        # 添加任务名称列方便回溯
        df_fc["task"] = entry
        df_fi["task"] = entry

        all_fc.append(df_fc)
        all_fi.append(df_fi)

    # 合并全部任务
    df_fc_all = pd.concat(all_fc, ignore_index=True) if all_fc else pd.DataFrame()
    df_fi_all = pd.concat(all_fi, ignore_index=True) if all_fi else pd.DataFrame()

    return df_fc_all, df_fi_all

# 统计组合出现情况 / Count feature combos occurrences
def count_combos(df_fc_all, df_fi_all=None):
    # 1. 统一“组合”的 key：把表达式排序后做成 tuple
    #    （这样 (A,B) 和 (B,A) 会被视作同一组；如果你不要这个行为，可以去掉 sorted）
    df_fc_all = df_fc_all.copy()
    df_fc_all["combo_key"] = df_fc_all["Feature_exprs"].apply(
        lambda exprs: tuple(sorted(exprs))
    )
    
    stats = {}
    for _, row in df_fc_all.iterrows():
        combo = row['combo_key']
        rank = row.get('Rank', None)
        rmse = row.get('RMSE')
        if combo not in stats:
            stats[combo] = {"count":0, "ranks": [], "RMSE": []}
        stats[combo]["count"] += 1
        # if '(DF/DI)' in combo:
        #     print(rank, row.get('task'))
        stats[combo]["ranks"].append((int(rank)))
        stats[combo]["RMSE"].append(float(rmse))
    rows = []


    # 3. 把组合展开成一行一条记录
    rows = []
    task_list = [str(r) for r in df_fc_all[df_fc_all["combo_key"] == combo]["task"]]
    for combo, info in stats.items():
        mask = df_fc_all["combo_key"] == combo
        tasks = df_fc_all.loc[mask, "task"].astype(str).tolist()
        ranks = [int(r) for r in df_fc_all.loc[mask, "Rank"]]
        rmse_list = [float(r) for r in df_fc_all.loc[mask, "RMSE"]]
        if df_fi_all is not None and not df_fi_all.empty:
            coeff_rows = df_fi_all.loc[mask, [c for c in df_fi_all.columns if c.startswith("c")]]
            coeffs = coeff_rows.apply(lambda row: f"({','.join(str(row[c]) for c in coeff_rows.columns)})", axis=1).tolist()
        else:
            coeffs = []

        rows.append({
            "features": " -- ".join(combo),
            "count": len(ranks),
            "ranks": "[" + ", ".join(str(r) for r in ranks) + "]",
            "mean_rank": sum(ranks) / len(ranks) if ranks else None,
            "RMSEs": "[" + ", ".join(f"{r:.3f}" for r in rmse_list) + "]",
            "mean_RMSE": sum(rmse_list) / len(rmse_list) if rmse_list else None,
            "task_list": "[" + ", ".join(tasks) + "]",
            "coeffs_list": "[" + ", ".join(coeffs) + "]",
        })

    df_combo = pd.DataFrame(rows).sort_values(
        by=["count","mean_rank"], ascending=[False,True]
    ).reset_index(drop=True)

    return df_combo
import argparse

# 主流程：汇总各任务结果并输出 
# Main flow: aggregate tasks and export
def main():
    parser = argparse.ArgumentParser(description="Collect SISSO results from multiple runs.")
    parser.add_argument("root_dir", type=str, help="Root directory containing SISSO run subdirectories.")
    parser.add_argument("--output","-o", type=str, default="feature_combination_stats.csv", help="Output CSV file for feature combination statistics.")
    args = parser.parse_args()
    df_fc_all, df_fi_all = collect_all_runs(args.root_dir)
    df_combo = count_combos(df_fc_all, df_fi_all)
    # 输出结果到文件
    # df_fc_all.to_csv("all_feature_combinations.csv", index=False)
    # df_fi_all.to_csv("all_feature_coefficients.csv", index=False)
    df_combo.to_csv(f"{args.output}.csv", index=False)


if __name__ == "__main__":
    main()
