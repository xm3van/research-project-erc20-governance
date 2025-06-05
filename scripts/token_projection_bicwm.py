import pandas as pd
import numpy as np
import networkx as nx
import os
from os.path import join
from dotenv import load_dotenv
from NEMtropy import BipartiteGraph

# Load environment variables
load_dotenv()
path = os.environ['DATA_DIRECTORY']

# File paths
SNAPSHOT_CSV_PATH = 'data/snapshot_selection.csv'
ADDRESS_CSV_PATH = 'data/final_token_selection.csv'
OUTPUT_PATH = join(path, 'data/validated_token_projection_graphs_biwcm')

KNOWN_BURNER_ADDRESSES = [
    '0x0000000000000000000000000000000000000000', '0x000000000000000000000000000000000000dead',
    '0x0000000000000000000000000000000000000001', '0x0000000000000000000000000000000000000002',
    '0x0000000000000000000000000000000000000003', '0x0000000000000000000000000000000000000004',
    '0x0000000000000000000000000000000000000005', '0x0000000000000000000000000000000000000006',
    '0x0000000000000000000000000000000000000007'
]

def matrix_to_nx_graph(adj_matrix: np.ndarray, row_labels: list, col_labels: list = None, directed: bool = False, threshold: float = 0.0, include_self_loops: bool = False) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()
    is_square = col_labels is None or row_labels == col_labels
    if is_square:
        col_labels = row_labels
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if not include_self_loops and is_square and i == j:
                continue
            weight = adj_matrix[i, j]
            if weight > threshold:
                G.add_edge(row_labels[i], col_labels[j], weight=weight)
    return G

def generate_network_graphs():
    df_snapshot = pd.read_csv(SNAPSHOT_CSV_PATH)
    df_addresses = pd.read_csv(ADDRESS_CSV_PATH)

    for snapshot in df_snapshot[df_snapshot['Block Height'] >= 11659570]['Block Height']:
        print(f"Processing snapshot {snapshot}...")
        ddf = pd.read_csv(join(path, f'data/snapshot_token_balance_tables_enriched/token_holder_snapshot_balance_labelled_{snapshot}.csv'))

        ddf = ddf[(ddf['pct_supply'] > 0.00001) &
                  (~ddf['address'].isin(KNOWN_BURNER_ADDRESSES)) &
                  (ddf['token_address'].isin(df_addresses['address']))]

        pivot_df = ddf.pivot_table(index='address', columns='token_address', values='pct_supply', fill_value=0)
        

        # Optional: Normalize each column to mitigate token scale differences
        # pivot_df = pivot_df.div(pivot_df.max(axis=0).replace(0, 1), axis=1)

        # Remove empty rows and columns directly in pivot_df (safer label alignment)
        nonzero_rows = (pivot_df != 0).any(axis=1)
        nonzero_cols = (pivot_df != 0).any(axis=0)

        # Filter pivot_df
        pivot_df = pivot_df.loc[nonzero_rows, nonzero_cols]

        # Convert to numpy and get aligned labels
        B = pivot_df.to_numpy()
        address_labels = pivot_df.index.tolist()
        token_labels = pivot_df.columns.tolist()



        print(f"Cleaned weighted matrix shape: {B.shape}")

        # Run BiWCM
        myGraph = BipartiteGraph(biadjacency=B)

        print('Starting "solve_tool" for BiWCM...')
        myGraph.solve_tool(
            model="biwcm",
            method='quasinewton',         # fixed-point is stable, and less memory-heavy than newton
            initial_guess='random',      # faster to converge than 'random' in most real-world cases
            max_steps=None,               # limit number of iterations to prevent stalling
            # tolerance=1e-4,               # relax convergence threshold (default is often 1e-6 or tighter)
            linsearch=False,              # disable line search to reduce per-step cost
            regularise=True,              # keeps solution bounded
            print_error=True
        )

        print('Computing validated matrix...')
        validated_bipartite = myGraph.get_validated_matrix(
            significance=0.01,
            validation_method='bonferroni',
        )

        print("Validated bipartite matrix shape:", validated_bipartite.shape)
        print("Nonzero entries:", np.count_nonzero(validated_bipartite))
        print("Token strength distribution (post-filtering):", validated_bipartite.sum(axis=0))
        print("Address strength distribution (post-filtering):", validated_bipartite.sum(axis=1))

        cooccur = validated_bipartite.T @ validated_bipartite
        np.fill_diagonal(cooccur, 0)

        print("Co-occurrence stats: min =", cooccur.min(), "max =", cooccur.max(), "mean =", cooccur.mean())

        token_token_projection = (cooccur > 0).astype(int)
        if np.count_nonzero(token_token_projection) == 0:
            print(f"❌ Snapshot {snapshot}: No validated token-token links found.")
            continue

        G = matrix_to_nx_graph(cooccur, token_labels, threshold=0.0)
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        out_file = join(OUTPUT_PATH, f'validated_token_projection_graph_{snapshot}.graphml')
        nx.write_graphml(G, out_file)
        print(f"✅ Saved GraphML for snapshot {snapshot} to {out_file}")

if __name__ == "__main__":
    generate_network_graphs()
