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
OUTPUT_PATH = join(path, 'data/validated_token_projection_graphs_nemtropy')

# Known burner addresses
KNOWN_BURNER_ADDRESSES = [
    '0x0000000000000000000000000000000000000000', '0x000000000000000000000000000000000000dead',
    '0x0000000000000000000000000000000000000001', '0x0000000000000000000000000000000000000002',
    '0x0000000000000000000000000000000000000003', '0x0000000000000000000000000000000000000004',
    '0x0000000000000000000000000000000000000005', '0x0000000000000000000000000000000000000006',
    '0x0000000000000000000000000000000000000007'
]

def matrix_to_nx_graph(
    adj_matrix: np.ndarray,
    row_labels: list,
    col_labels: list = None,
    directed: bool = False,
    threshold: float = 0.0,
    include_self_loops: bool = False
) -> nx.Graph:
    """
    Converts an adjacency or bipartite matrix to a NetworkX graph.

    Parameters:
    - adj_matrix (np.ndarray): The adjacency or bipartite matrix.
    - row_labels (list): Labels for the row nodes.
    - col_labels (list): Labels for the column nodes. If None, assumes square matrix.
    - directed (bool): Whether to return a directed graph.
    - threshold (float): Only include edges with weight > threshold.
    - include_self_loops (bool): Whether to include self-loops in square matrices.

    Returns:
    - G (nx.Graph or nx.DiGraph): The resulting graph.
    """
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

        ddf = ddf[(ddf['value'] > 0) &
                  (~ddf['address'].isin(KNOWN_BURNER_ADDRESSES)) &
                  (ddf['token_address'].isin(df_addresses['address']))]

        # Pivot to binary address-token matrix
        pivot_df = ddf.assign(value=1).pivot_table(index='address', columns='token_address', values='value', fill_value=0)

        # print(pivot_df.head(3))
                # Convert to matrix
        B = pivot_df.to_numpy()

        # Step 1: Remove empty rows and columns (disconnected nodes)
        nonzero_rows = ~np.all(B == 0, axis=1)
        nonzero_cols = ~np.all(B == 0, axis=0)

        B = B[nonzero_rows][:, nonzero_cols]

        # Update labels to match filtered matrix
        address_labels = pivot_df.index[nonzero_rows].tolist()
        token_labels = pivot_df.columns[nonzero_cols].tolist()

        # Step 2: Remove duplicate rows (identical addresses)
        _, unique_row_indices = np.unique(B, axis=0, return_index=True)
        B = B[sorted(unique_row_indices)]
        address_labels = [address_labels[i] for i in sorted(unique_row_indices)]

        # Step 3 (Optional): Log dimensions for debug
        print(f"Cleaned matrix shape: {B.shape}")


        # Create and solve BiCM model
        myGraph = BipartiteGraph(biadjacency=B)

        print('Starting "solve_tool"')

        myGraph.solve_tool(
            model="bicm",
            light_mode=False,
            method='fixed-point',
            initial_guess='random',
            max_steps=None,
            verbose=False,
            linsearch=True,
            regularise=True,
            print_error=True
        )

        print('Starting "get_validated_matrix"')

        # Statistically validated projection (Bonferroni-corrected)
        validated_bipartite = myGraph.get_validated_matrix(
            significance=0.01,
            validation_method='bonferroni', 
        )

        print("Validated bipartite matrix shape:", validated_bipartite.shape)
        print("Nonzero entries:", np.count_nonzero(validated_bipartite))
        print("Token degree distribution (post-filtering):", validated_bipartite.sum(axis=0))
        print("Address degree distribution (post-filtering):", validated_bipartite.sum(axis=1))

        cooccur = validated_bipartite.T @ validated_bipartite
        np.fill_diagonal(cooccur, 0)
        print("Co-occurrence stats:")
        print("Min:", cooccur.min(), "Max:", cooccur.max(), "Mean:", cooccur.mean())



        # Compute token-token co-occurrence matrix (projection)
        token_token_projection = validated_bipartite.T @ validated_bipartite
        token_token_projection = (token_token_projection > 0).astype(int)

        np.fill_diagonal(token_token_projection, 0)  # optional: remove self-loops


        if np.count_nonzero(token_token_projection) == 0:
            print(f"❌ Snapshot {snapshot}: No validated token-token links found.")
            continue

        # Convert to networkx graph
        G = matrix_to_nx_graph(token_token_projection, token_labels)

        # Save GraphML
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        out_file = join(OUTPUT_PATH, f'validated_token_projection_graph_{snapshot}.graphml')
        nx.write_graphml(G, out_file)
        print(f"Saved GraphML for snapshot {snapshot} to {out_file}")

if __name__ == "__main__":
    generate_network_graphs()
