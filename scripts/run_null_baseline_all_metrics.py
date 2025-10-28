#!/usr/bin/env python3
"""
Null baseline for ALL link-level metrics produced by LinkAnalysis.

• Keeps empirical validated links (SVN output).
• Randomizes balances WITHIN each token (preserves token marginals, breaks cross-token correlation).
• Recomputes all metrics for each permutation.
• Outputs per-snapshot CSV with empirical value, null mean/std, percentile, z-score.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from os.path import join
from dotenv import load_dotenv
from tqdm import tqdm

from src.analysis.link_analysis import LinkAnalysis
from src.utilities.utils import *
# from src.utilities.metrics_and_tests import *  # if needed elsewhere

load_dotenv()
DATA_DIR = '.'

SNAPSHOT_SELECTION = "data/snapshot_selection.csv"
TOKEN_LIST        = "data/final_token_selection.csv"
PRICE_TABLE       = "data/price_table.csv"                    # <- like links_main
ENRICHED_DIR      = join(DATA_DIR, "data/snapshot_token_balance_tables_enriched")
VALIDATED_DIR     = join(DATA_DIR, "data/validated_token_projection_graphs")
OUT_DIR           = join(DATA_DIR, "output/links/null_baseline_all_metrics")
os.makedirs(OUT_DIR, exist_ok=True)


KNOWN_BURNERS = {'0x0000000000000000000000000000000000000000',
                          '0x0000000000000000000000000000000000000000',
                          '0x0000000000000000000000000000000000000001',
                          '0x0000000000000000000000000000000000000002',
                          '0x0000000000000000000000000000000000000003',
                          '0x0000000000000000000000000000000000000004',
                          '0x0000000000000000000000000000000000000005',
                          '0x0000000000000000000000000000000000000006',
                          '0x0000000000000000000000000000000000000007',
                          '0x000000000000000000000000000000000000dead'}

N_ITER = 10_000
SEED   = 42

def add_value_usd(df_bal: pd.DataFrame, price_table: pd.DataFrame, block_height: int) -> pd.DataFrame:
    """Ensure 'value_usd' exists (match your links_main logic)."""
    # price_table is indexed by token_address (lower) and columns are block heights as strings
    prices = price_table[str(block_height)]
    df_bal = df_bal.copy()
    df_bal["token_address"] = df_bal["token_address"].str.lower()
    df_bal["token_price_usd"] = df_bal["token_address"].map(prices.to_dict())
    # your enriched tables store 'value' in wei; adjust if different in your data
    df_bal["value_usd"] = (df_bal["value"] / (10**18)) * df_bal["token_price_usd"]
    return df_bal

def get_link_members(df_bal: pd.DataFrame, link_tokens: list[str]) -> list[str]:
    """Replicates LinkAnalysis.link_member_wallets() but avoids constructing the class."""
    members = set(df_bal[df_bal.token_address == link_tokens[0]].address.unique())
    for t in link_tokens[1:]:
        addr = set(df_bal[df_bal.token_address == t].address.unique())
        members &= addr
        if not members:
            return []
    return list(members)

def build_subframes_for_link(df_bal: pd.DataFrame, link_tokens: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct sub_dataFrame and sample_population consistent with your pipeline."""
    members = get_link_members(df_bal, link_tokens)
    if not members:
        return None, None
    sub_df = df_bal[df_bal.address.isin(members)].copy()
    ctrl_df = df_bal.copy()  # sample population is whole snapshot (your current approach)
    return sub_df, ctrl_df

def permute_within_token(df_bal: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle 'value' across addresses within each token (preserve token totals)."""
    def _permute(g):
        vals = g["value"].to_numpy()
        rng.shuffle(vals)
        g = g.copy()
        g["value"] = vals
        return g
    out = df_bal.groupby("token_address", group_keys=False).apply(_permute)
    # IMPORTANT: recompute value_usd after value changes
    if "token_price_usd" in out.columns:
        out["value_usd"] = (out["value"] / (10**18)) * out["token_price_usd"]
    return out

def compute_empirical_metrics(df_bal: pd.DataFrame, G: nx.Graph, token_lookup: dict) -> dict:
    """Compute all non-directional metrics for each link using your LinkAnalysis class."""
    results = {}
    edges = [tuple(sorted(e)) for e in G.edges()]
    for (u, v) in edges:
        sub_df, ctrl_df = build_subframes_for_link(df_bal, [u, v])
        if sub_df is None:
            continue
        la = LinkAnalysis([u, v], df_bal, sub_df, ctrl_df, token_lookup)
        la.directional = False
        link_name, res, _, _ = la.analyze_link()
        # standardize a link identifier that is easy to join on
        link_id = f"({token_lookup[u]},{token_lookup[v]})"
        results[link_id] = res
    return results

def run_snapshot(bh: int,
                 df_bal_raw: pd.DataFrame,
                 G: nx.Graph,
                 token_lookup: dict,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Run empirical + null on one snapshot; return long table of results."""
    # empirical (ensure value_usd)
    df_emp = compute_empirical_metrics(df_bal_raw, G, token_lookup)
    if not df_emp:
        return pd.DataFrame()

    # collect metric names
    any_metrics = next(iter(df_emp.values()))
    metric_names = list(any_metrics.keys())

    # init null containers: metric -> link -> list
    links = list(df_emp.keys())
    null = {m: {L: [] for L in links} for m in metric_names}

    # permutations
    for _ in tqdm(range(N_ITER), desc=f"BH {bh} permutations"):
        df_perm = permute_within_token(df_bal_raw, rng)
        df_perm_metrics = compute_empirical_metrics(df_perm, G, token_lookup)
        # only accumulate for links that existed empirically
        for L in links:
            mvals = df_perm_metrics.get(L)
            if mvals is None:
                # if permuted breaks membership (rare), append NaN to keep lengths aligned
                for m in metric_names:
                    null[m][L].append(np.nan)
            else:
                for m in metric_names:
                    null[m][L].append(mvals.get(m, np.nan))

    # summarize to long dataframe
    rows = []
    for L, emp in df_emp.items():
        sym_u, sym_v = L.strip("()").split(",")
        for m in metric_names:
            dist = np.array(null[m][L], dtype=float)
            dist = dist[~np.isnan(dist)]
            if dist.size == 0:
                mu = sd = q = z = np.nan
            else:
                mu = float(np.mean(dist))
                sd = float(np.std(dist, ddof=1)) if dist.size > 1 else 0.0
                q  = float((dist < emp[m]).mean())
                z  = float((emp[m] - mu) / sd) if sd > 0 else np.nan
            rows.append({
                "snapshot_block": bh,
                "link": L,
                "token_u": sym_u,
                "token_v": sym_v,
                "metric": m,
                "empirical_value": float(emp[m]) if emp[m] is not None else np.nan,
                "null_mean": mu,
                "null_std": sd,
                "quantile": q,
                "zscore": z
            })
    return pd.DataFrame(rows)

def main():
    rng = np.random.default_rng(SEED)
    df_snap   = pd.read_csv(SNAPSHOT_SELECTION)
    df_tokens = pd.read_csv(TOKEN_LIST)
    df_price  = pd.read_csv(PRICE_TABLE, index_col=0)   # token rows, block-height columns (string)
    df_price.index = df_price.index.str.lower()

    token_lookup = dict(zip(df_tokens["address"].str.lower(), df_tokens["symbol"]))

    for _, row in df_snap.iterrows():
        bh = int(row["Block Height"])
        snap_file = join(ENRICHED_DIR, f"token_holder_snapshot_balance_labelled_{bh}.csv")
        graph_file = join(VALIDATED_DIR, f"validated_token_projection_graph_{bh}.graphml")
        if not os.path.exists(snap_file) or not os.path.exists(graph_file):
            print(f"Skipping {bh}: missing {os.path.basename(snap_file)} or {os.path.basename(graph_file)}")
            continue

        df_bal = pd.read_csv(snap_file)
        df_bal = df_bal[~df_bal["address"].isin(KNOWN_BURNERS)].copy()
        df_bal["token_address"] = df_bal["token_address"].str.lower()
        # ensure value_usd exists (and keep pct_supply as in your tables)
        df_bal = add_value_usd(df_bal, df_price, bh)

        G = nx.read_graphml(graph_file)
        out = run_snapshot(bh, df_bal, G, token_lookup, rng)
        if out.empty:
            print(f"No links for {bh}; continuing.")
            continue

        out_path = join(OUT_DIR, f"null_baseline_all_metrics_{bh}.csv")
        out.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
