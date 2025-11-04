#!/usr/bin/env python3
from __future__ import annotations
import os
from os.path import join
import numpy as np
import pandas as pd
import networkx as nx
from dotenv import load_dotenv
from pandas.api.types import CategoricalDtype

# ============================================================
# CONFIG
# ============================================================
N_ITER = 10_000            # bump high (e.g., 5000-10000) for final results
RANDOM_STATE = 42
OUT_DIR = "output/links/null_model"
TABLES_SUBDIR = "tables"
FIXED_LABELS = None       # e.g. ["cex", "lending", "dao", "unknown"]

# --- NEW: persistence filters for reporting ---
MIN_SIG_COUNT = 9         # keep links significant at least this many times
MIN_SNAPSHOT_COUNT = None # e.g., 6 to avoid tiny-N artefacts; None disables

# metrics we *always* compute
NULL_METRICS = [
    "link_size",
    "avg_token_holding_share",
    "dir_share_token_a",
    "dir_share_token_b",
    "median_token_value_usd",
    "gini_token_holding_share",
]

# metrics we KEEP in the significance table and for LaTeX
SIG_METRICS_ALLOWLIST = {
    "avg_token_holding_share",
    "dir_share_token_a",
    "dir_share_token_b",
    "gini_token_holding_share",
    "median_token_value_usd",
}

# nice names for LaTeX captions
METRIC_LABELS = {
    "avg_token_holding_share": "Average Token Holding Share",
    "dir_share_token_a": "Directional Token Holding Share (token A)",
    "dir_share_token_b": "Directional Token Holding Share (token B)",
    "gini_token_holding_share": "Token Holding Share Inequality (Gini)",
    "median_token_value_usd": "Median Wealth of Link",
}

# how many rows we show per LaTeX table
MAX_ROWS_TEX = 200

# ============================================================
# HELPERS
# ============================================================
def to_category(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, CategoricalDtype):
        return series
    return series.astype("category")

def gini(x: np.ndarray) -> float:
    x = x.astype(float)
    if x.size == 0 or np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return (np.sum((2 * idx - n - 1) * x)) / (n * np.sum(x))

def build_numeric_snapshot(df_bal: pd.DataFrame) -> dict:
    addr_cat = to_category(df_bal["address"])
    tok_cat = to_category(df_bal["token_address"])

    df_bal = df_bal.copy()
    df_bal["addr_id"] = addr_cat.cat.codes.to_numpy()
    df_bal["tok_id"] = tok_cat.cat.codes.to_numpy()

    return {
        "df_bal": df_bal,
        "addr_id": df_bal["addr_id"].to_numpy(),
        "tok_id": df_bal["tok_id"].to_numpy(),
        "pct": df_bal["pct_supply"].to_numpy(),
        "val_usd": df_bal["value_usd"].to_numpy(),
        "addr_count": len(addr_cat.cat.categories),
        "tok_count": len(tok_cat.cat.categories),
        "addr_cat": addr_cat,
        "tok_cat": tok_cat,
    }

def build_links(graph_path: str, num: dict):
    G = nx.read_graphml(graph_path)
    tok_cat: pd.Series = to_category(num["tok_cat"])
    tok_str_to_code = {t: i for i, t in enumerate(tok_cat.cat.categories)}
    tok_code_to_str = {i: t for t, i in tok_str_to_code.items()}
    link_pairs = [tuple(sorted((str(u), str(v)))) for u, v in G.edges()]
    return link_pairs, tok_str_to_code, tok_code_to_str

def compute_pvalue(observed: float, null_samples: np.ndarray) -> float:
    return float((null_samples >= observed).mean())

def compute_metrics_for_link(
    link_addr_mask: np.ndarray,
    tok_a_code: int,
    tok_b_code: int,
    addr_token_pct: dict[int, np.ndarray],
    addr_token_val: dict[int, np.ndarray],
    num: dict,
    label_col: str | None,
) -> dict:
    df_bal_num = num["df_bal"]
    idx = np.nonzero(link_addr_mask)[0]

    # 1) link size
    link_size = float(idx.size)

    if idx.size == 0:
        return {
            "link_size": 0.0,
            "avg_token_holding_share": 0.0,
            "dir_share_token_a": 0.0,
            "dir_share_token_b": 0.0,
            "median_token_value_usd": 0.0,
            "gini_token_holding_share": 0.0,
            "label_shares": {},
        }

    # 2) avg token holding share
    a_share = addr_token_pct[tok_a_code][idx]
    b_share = addr_token_pct[tok_b_code][idx]
    per_addr_avg_share = 0.5 * (a_share + b_share)
    avg_token_holding_share = float(per_addr_avg_share.sum())

    # 3) directional
    dir_share_token_a = float(a_share.sum())
    dir_share_token_b = float(b_share.sum())

    # 4) median USD
    vals_a = addr_token_val[tok_a_code][idx]
    vals_b = addr_token_val[tok_b_code][idx]
    total_vals = vals_a + vals_b
    median_token_value_usd = float(np.median(total_vals))

    # 5) gini
    gini_token_holding_share = float(gini(per_addr_avg_share))

    # 6) labels
    label_shares = {}
    if label_col is not None and label_col in df_bal_num.columns:
        addr_label_map = (
            df_bal_num[["addr_id", label_col]]
            .dropna(subset=[label_col])
            .drop_duplicates(subset=["addr_id"])
            .set_index("addr_id")[label_col]
            .to_dict()
        )
        labels_to_use = set(addr_label_map.values()) if FIXED_LABELS is None else set(FIXED_LABELS)
        denom = per_addr_avg_share.sum()
        if denom > 0:
            for a_id, share_val in zip(idx, per_addr_avg_share):
                lab = addr_label_map.get(int(a_id))
                if lab in labels_to_use:
                    label_shares[lab] = label_shares.get(lab, 0.0) + float(share_val)
            for lab in list(label_shares.keys()):
                label_shares[lab] = label_shares[lab] / float(denom)

    return {
        "link_size": link_size,
        "avg_token_holding_share": avg_token_holding_share,
        "dir_share_token_a": dir_share_token_a,
        "dir_share_token_b": dir_share_token_b,
        "median_token_value_usd": median_token_value_usd,
        "gini_token_holding_share": gini_token_holding_share,
        "label_shares": label_shares,
    }

# ============================================================
# POST-PROCESSING FOR APPENDIX TABLES
# ============================================================
def aggregate_for_appendix(df_all: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build per-metric summary across ALL snapshots, with persistence filtering.
    Returns dict: metric -> aggregated & filtered DataFrame
    """
    out = {}
    for metric in SIG_METRICS_ALLOWLIST:
        if metric not in df_all["metric"].unique():
            continue
        dfm = df_all[df_all["metric"] == metric].copy()
        gb = dfm.groupby(["token_a", "token_b", "link_name"], dropna=False)

        rows = []
        for (tok_a, tok_b, link_name), g in gb:
            n_snap = len(g)
            n_sig = int((g["p_value"] <= 0.05).sum())
            pct_sig = n_sig / n_snap if n_snap else 0.0
            median_obs = pd.to_numeric(g["observed"], errors="coerce").median()
            median_p95 = pd.to_numeric(g["p95"], errors="coerce").median()
            rows.append(
                {
                    "token_a": tok_a,
                    "token_b": tok_b,
                    "link_name": link_name,
                    "n_snapshots": n_snap,
                    "n_significant": n_sig,
                    "pct_significant": pct_sig,
                    "median_observed": median_obs,
                    "median_null_p95": median_p95,
                }
            )

        df_out = pd.DataFrame(rows)

        # --- NEW: apply persistence filters ---
        if MIN_SNAPSHOT_COUNT is not None:
            df_out = df_out[df_out["n_snapshots"] >= int(MIN_SNAPSHOT_COUNT)]
        if MIN_SIG_COUNT is not None:
            df_out = df_out[df_out["n_significant"] >= int(MIN_SIG_COUNT)]

        # sort: most persistent & strongest first
        df_out = df_out.sort_values(
            by=["n_significant", "pct_significant", "median_observed"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        out[metric] = df_out
    return out

def _latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s

def write_latex_tables(per_metric: dict[str, pd.DataFrame], base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    for metric, df in per_metric.items():
        df_print = df.copy()
        if len(df_print) > MAX_ROWS_TEX:
            df_print = df_print.head(MAX_ROWS_TEX)

        df_print = df_print[[
            "link_name", "n_snapshots", "n_significant",
            "pct_significant", "median_observed", "median_null_p95",
        ]].copy()

        df_print["pct_significant"] = (pd.to_numeric(df_print["pct_significant"], errors="coerce") * 100).round(1)
        df_print["median_observed"] = pd.to_numeric(df_print["median_observed"], errors="coerce").round(4)
        df_print["median_null_p95"] = pd.to_numeric(df_print["median_null_p95"], errors="coerce").round(4)
        df_print = df_print.fillna("--")
        df_print["link_name"] = df_print["link_name"].map(_latex_escape)

        headers = ["Link", "N", "Sig.", r"\% Sig.", "Median Obs.", "Median Null p95"]
        colspec = "lrrrrr"

        lines = []
        lines.append(r"\begin{table}[!ht]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(rf"\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\hline")
        lines.append(" & ".join(headers) + r" \\")
        lines.append(r"\hline")

        for _, r in df_print.iterrows():
            row = [
                str(r["link_name"]),
                str(int(r["n_snapshots"])) if r["n_snapshots"] != "--" else "--",
                str(int(r["n_significant"])) if r["n_significant"] != "--" else "--",
                f"{r['pct_significant']}" if r["pct_significant"] != "--" else "--",
                f"{r['median_observed']}" if r["median_observed"] != "--" else "--",
                f"{r['median_null_p95']}" if r["median_null_p95"] != "--" else "--",
            ]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"}")  # end resizebox

        nice = METRIC_LABELS.get(metric, metric)
        cap = (
            f"Appendix – {nice} (significance persistence; "
            f"min. sig. occurrences = {MIN_SIG_COUNT}"
            + (f", min. snapshots = {MIN_SNAPSHOT_COUNT}" if MIN_SNAPSHOT_COUNT else "")
            + "). A link is counted as significant in a snapshot if the observed value exceeds the "
              "95th percentile of the null model (empirical p \\(\\le 0.05\\))."
        )
        lines.append(rf"\vspace{{2mm}}\textit{{{_latex_escape(cap)}}}")
        lines.append(r"\end{table}")

        tex = "\n".join(lines)
        suffix = f"_minsig{MIN_SIG_COUNT}" + (f"_minsnap{MIN_SNAPSHOT_COUNT}" if MIN_SNAPSHOT_COUNT else "")
        out_path = join(base_dir, f"appendix_table_{metric}{suffix}.tex")
        with open(out_path, "w") as f:
            f.write(tex)

# ============================================================
# MAIN
# ============================================================
def main():
    load_dotenv()
    DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY", ".")
    os.makedirs(join(DATA_DIRECTORY, OUT_DIR), exist_ok=True)

    df_snapshots = pd.read_csv("data/snapshot_selection.csv")
    df_tokens = pd.read_csv("data/final_token_selection.csv")

    # address → symbol map (lowercased)
    token_sym_map = {
        str(row["address"]).lower(): str(row.get("symbol", row["address"]))
        for _, row in df_tokens.iterrows()
    }

    rng = np.random.default_rng(RANDOM_STATE)
    all_full = []
    all_sig = []

    for _, row in df_snapshots.iterrows():
        snap_block = int(row["Block Height"])
        snap_date = str(row["Date"])
        print(f"\n▶ snapshot {snap_date} @ {snap_block}")

        bal_path = join(
            DATA_DIRECTORY,
            "data/snapshot_token_balance_tables_enriched",
            f"token_holder_snapshot_balance_labelled_{snap_block}.csv",
        )
        if not os.path.exists(bal_path):
            print(f"  ⚠️ skip {snap_block}: balance file missing")
            continue

        df_bal = pd.read_csv(bal_path)
        df_bal = df_bal[df_bal.token_address.str.lower().isin(df_tokens.address.str.lower())].copy()
        if df_bal.empty:
            print(f"  ⚠️ skip {snap_block}: no tokens from universe")
            continue

        if "value_usd" not in df_bal.columns:
            price_path = "data/price_table.csv"
            if not os.path.exists(price_path):
                print(f"  ⚠️ skip {snap_block}: price table missing")
                continue
            df_price = pd.read_csv(price_path, index_col=0)
            if str(snap_block) not in df_price.columns:
                print(f"  ⚠️ skip {snap_block}: no price col for {snap_block}")
                continue
            df_bal["token_price_usd"] = df_bal["token_address"].apply(
                lambda x: df_price.loc[str(x), str(snap_block)]
            )
            df_bal["value_usd"] = df_bal["value"] / 1e18 * df_bal["token_price_usd"]

        burners = {
            "0x0000000000000000000000000000000000000000",
            "0x0000000000000000000000000000000000000001",
            "0x0000000000000000000000000000000000000002",
            "0x0000000000000000000000000000000000000003",
            "0x0000000000000000000000000000000000000004",
            "0x0000000000000000000000000000000000000005",
            "0x0000000000000000000000000000000000000006",
            "0x0000000000000000000000000000000000000007",
            "0x000000000000000000000000000000000000dead",
        }
        df_bal = df_bal[~df_bal.address.isin(burners)].copy()

        num = build_numeric_snapshot(df_bal)

        gpath = join(
            DATA_DIRECTORY,
            "data/validated_token_projection_graphs",
            f"validated_token_projection_graph_{snap_block}.graphml",
        )
        if not os.path.exists(gpath):
            print(f"  ⚠️ skip {snap_block}: graph file missing")
            continue

        link_pairs, tok_str_to_code, _ = build_links(gpath, num)
        if not link_pairs:
            print(f"  ⚠️ skip {snap_block}: no links")
            continue

        addr_id = num["addr_id"]
        tok_id = num["tok_id"]
        pct = num["pct"]
        val_usd = num["val_usd"]
        addr_count = num["addr_count"]
        df_bal_num = num["df_bal"]

        rows_per_token = {t: np.where(tok_id == t)[0] for t in np.unique(tok_id)}

        # observed per-token
        addr_token_pct_obs = {}
        addr_token_val_obs = {}
        for t_code, idx_rows in rows_per_token.items():
            arr_pct = np.zeros(addr_count, dtype=np.float32)
            arr_val = np.zeros(addr_count, dtype=np.float32)
            np.add.at(arr_pct, addr_id[idx_rows], pct[idx_rows])
            np.add.at(arr_val, addr_id[idx_rows], val_usd[idx_rows])
            addr_token_pct_obs[t_code] = arr_pct
            addr_token_val_obs[t_code] = arr_val

        label_col = "label" if "label" in df_bal_num.columns else None

        # observed metrics
        observed_by_link = {}
        for (tok_a, tok_b) in link_pairs:
            a_code = tok_str_to_code[tok_a]
            b_code = tok_str_to_code[tok_b]
            mask_obs = (addr_token_pct_obs[a_code] > 0) & (addr_token_pct_obs[b_code] > 0)
            obs_metrics = compute_metrics_for_link(
                mask_obs,
                a_code,
                b_code,
                addr_token_pct_obs,
                addr_token_val_obs,
                num,
                label_col,
            )
            observed_by_link[(tok_a, tok_b)] = obs_metrics

        # null samples
        rng = np.random.default_rng(RANDOM_STATE)
        null_store = {pair: {m: [] for m in NULL_METRICS} for pair in link_pairs}

        for it in range(N_ITER):
            addr_token_pct = {}
            addr_token_val = {}
            for t_code, idx_rows in rows_per_token.items():
                n = idx_rows.size
                new_addrs = rng.choice(num["addr_count"], n, replace=(n > num["addr_count"]))
                pct_vals = pct[idx_rows].copy()
                val_vals = val_usd[idx_rows].copy()
                rng.shuffle(pct_vals)
                rng.shuffle(val_vals)

                arr_pct = np.zeros(num["addr_count"], dtype=np.float32)
                arr_val = np.zeros(num["addr_count"], dtype=np.float32)
                np.add.at(arr_pct, new_addrs, pct_vals)
                np.add.at(arr_val, new_addrs, val_vals)

                addr_token_pct[t_code] = arr_pct
                addr_token_val[t_code] = arr_val

            for (tok_a, tok_b) in link_pairs:
                a_code = tok_str_to_code[tok_a]
                b_code = tok_str_to_code[tok_b]
                mask = (addr_token_pct[a_code] > 0) & (addr_token_pct[b_code] > 0)
                mvals = compute_metrics_for_link(
                    mask,
                    a_code,
                    b_code,
                    addr_token_pct,
                    addr_token_val,
                    num,
                    None,
                )
                for m in NULL_METRICS:
                    null_store[(tok_a, tok_b)][m].append(mvals[m])

            if (it + 1) % 100 == 0:
                print(f"    iteration {it+1}/{N_ITER} done")

        # aggregate
        full_rows = []
        sig_rows = []
        for (tok_a, tok_b) in link_pairs:
            obs = observed_by_link[(tok_a, tok_b)]
            # symbols
            sym_a = token_sym_map.get(tok_a.lower(), tok_a)
            sym_b = token_sym_map.get(tok_b.lower(), tok_b)
            link_name = f"{sym_a}-{sym_b}"

            for m in NULL_METRICS:
                null_arr = np.asarray(null_store[(tok_a, tok_b)][m], dtype=np.float32)
                pval = compute_pvalue(obs[m], null_arr)
                rec = {
                    "snapshot_date": snap_date,
                    "snapshot_block": snap_block,
                    "token_a": tok_a,
                    "token_b": tok_b,
                    "link_name": link_name,
                    "metric": m,
                    "observed": obs[m],
                    "p01": float(np.percentile(null_arr, 1)),
                    "p05": float(np.percentile(null_arr, 5)),
                    "p10": float(np.percentile(null_arr, 10)),
                    "p25": float(np.percentile(null_arr, 25)),
                    "p50": float(np.percentile(null_arr, 50)),
                    "p75": float(np.percentile(null_arr, 75)),
                    "p90": float(np.percentile(null_arr, 90)),
                    "p95": float(np.percentile(null_arr, 95)),
                    "p99": float(np.percentile(null_arr, 99)),
                    "p_value": pval,
                    "n_iter": int(null_arr.size),
                }
                full_rows.append(rec)

                if (pval <= 0.05) and (m in SIG_METRICS_ALLOWLIST):
                    sig_rows.append(rec)

            if obs["label_shares"]:
                for lab, share_val in obs["label_shares"].items():
                    full_rows.append({
                        "snapshot_date": snap_date,
                        "snapshot_block": snap_block,
                        "token_a": tok_a,
                        "token_b": tok_b,
                        "link_name": link_name,
                        "metric": f"label_share__{lab}",
                        "observed": share_val,
                        "p01": np.nan, "p05": np.nan, "p10": np.nan, "p25": np.nan,
                        "p50": np.nan, "p75": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan,
                        "p_value": np.nan,
                        "n_iter": N_ITER,
                    })

        df_full = pd.DataFrame(full_rows)
        df_sig = pd.DataFrame(sig_rows)

        out_full = join(DATA_DIRECTORY, OUT_DIR, f"null_model_full_{snap_block}.csv")
        df_full.to_csv(out_full, index=False)
        print(f"  ✅ wrote {out_full}")

        if not df_sig.empty:
            out_sig = join(DATA_DIRECTORY, OUT_DIR, f"null_model_sig_{snap_block}.csv")
            df_sig.to_csv(out_sig, index=False)
            print(f"  ✅ wrote {out_sig}")

        all_full.append(df_full)
        if not df_sig.empty:
            all_sig.append(df_sig)

    # combine
    if all_full:
        df_all_full = pd.concat(all_full, ignore_index=True)
        combined_full_path = join(DATA_DIRECTORY, OUT_DIR, "null_model_full_all_snapshots.csv")
        df_all_full.to_csv(combined_full_path, index=False)
        print("✅ wrote combined full table")

        # build appendix tables (with persistence filters)
        per_metric = aggregate_for_appendix(df_all_full)
        tables_dir = join(DATA_DIRECTORY, OUT_DIR, TABLES_SUBDIR)
        write_latex_tables(per_metric, tables_dir)
        print(f"✅ wrote LaTeX tables to {tables_dir}")

        # also export filtered CSVs per metric for your own inspection
        for metric, df in per_metric.items():
            suffix = f"_minsig{MIN_SIG_COUNT}" + (f"_minsnap{MIN_SNAPSHOT_COUNT}" if MIN_SNAPSHOT_COUNT else "")
            df.to_csv(join(DATA_DIRECTORY, OUT_DIR, f"null_model_{metric}_summary{suffix}.csv"), index=False)

    if all_sig:
        df_all_sig = pd.concat(all_sig, ignore_index=True)
        df_all_sig.to_csv(join(DATA_DIRECTORY, OUT_DIR, "null_model_sig_all_snapshots.csv"), index=False)
        print("✅ wrote combined significant table")

if __name__ == "__main__":
    main()
