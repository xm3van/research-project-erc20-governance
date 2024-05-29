import random
import datetime
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from os.path import join
import os
import pickle
from dotenv import load_dotenv

from src.utilities.metrics_and_tests import *
from src.utilities.utils import *
from src.analysis.clique_analysis_class import CliqueAnalysis, clique_member_wallets_upper, clique_member_wallets_lower

load_dotenv()

# Load environment variables
path = os.environ['DATA_DIRECTORY']
TOKEN_BALANCE_TABLE_INPUT_PATH = join(path, "data/snapshot_token_balance_tables_enriched")
VALIDATED_PROJECTIONS_INPUT_PATH = join(path, 'data/validated_token_projection_graphs')

# Load datasets
df_snapshots = pd.read_csv('data/snapshot_selection.csv')
df_tokens = pd.read_csv("data/final_token_selection.csv")
df_token_price = pd.read_csv("data/price_table.csv", index_col=0)

# Known burner addresses
known_burner_addresses = [
    '0x0000000000000000000000000000000000000000',
    '0x0000000000000000000000000000000000000001',
    '0x0000000000000000000000000000000000000002',
    '0x0000000000000000000000000000000000000003',
    '0x0000000000000000000000000000000000000004',
    '0x0000000000000000000000000000000000000005',
    '0x0000000000000000000000000000000000000006',
    '0x0000000000000000000000000000000000000007',
    '0x000000000000000000000000000000000000dead'
]

def load_and_prepare_data(snapshot_block_height):
    ddf = pd.read_csv(join(TOKEN_BALANCE_TABLE_INPUT_PATH, f'token_holder_snapshot_balance_labelled_{snapshot_block_height}.csv'))
    ddf = ddf[~ddf.address.isin(known_burner_addresses)]
    ddf = ddf[ddf.pct_supply > 0.000005]

    ddf['token_price_usd'] = float

    for contract_address in ddf.token_address.unique():
        token_price = df_token_price.loc[str(contract_address), str(snapshot_block_height)]
        ddf.loc[ddf.token_address == contract_address, 'token_price_usd'] = token_price

    ddf['value_usd'] = (ddf['value'] / (10 ** 18) * ddf['token_price_usd'])

    return ddf

def get_cliques(G):
    token_lookup = df_tokens[['address', 'symbol']].set_index('address')['symbol'].to_dict()
    nx.set_node_attributes(G, token_lookup, 'name')

    all_cliques = [sorted(clique) for clique in nx.find_cliques(G) if len(clique) > 2]
    return all_cliques

def analyze_cliques(snapshot_block_height, snapshot_date, ddf, all_cliques, filter_method):
    token_lookup = df_tokens[['address', 'symbol']].set_index('address')['symbol'].to_dict()
    cliques_snapshot = {}
    cliques_snapshot_control = {}
    cliques_pvalues = {}

    for clique in all_cliques:
        clique_members_unique = []
        if filter_method == 'upper_bound':
            clique_members_unique = clique_member_wallets_upper(clique, ddf)
        elif filter_method == 'lower_bound':
            clique_members_unique = clique_member_wallets_lower(clique, ddf)

        if not clique_members_unique:
            print("Skip")
            continue

        ddf_sub = ddf[ddf.address.isin(clique_members_unique)].copy()
        filter1 = ddf.token_address.isin(clique)
        filter2 = ddf['address'].isin(clique_members_unique)
        relevant_population = list(ddf[(filter1 == True) & (filter2 != True)].address.unique())
        control_sample = random.sample(relevant_population, len(clique_members_unique))
        ddf_sub_control = ddf[ddf.address.isin(control_sample)].copy()

        analyzer = CliqueAnalysis(ddf_sub, ddf_sub_control, ddf, clique, token_lookup)
        clique_name, results, results_control, pvalues = analyzer.analyze()

        cliques_snapshot[str(clique_name)] = results
        cliques_snapshot_control[str(clique_name)] = results_control
        cliques_pvalues[str(clique_name)] = pvalues

    return cliques_snapshot, cliques_snapshot_control, cliques_pvalues

def cliques_main():
    cliques = {
        'upper_bound': {'sample': {}, 'control': {}, 'pvalues': {}},
        'lower_bound': {'sample': {}, 'control': {}, 'pvalues': {}}
    }

    for _, row in df_snapshots[df_snapshots['Block Height'] > 11547458].iterrows():
        snapshot_date = row['Date']
        snapshot_block_height = row['Block Height']

        print(f"Snapshot for Block Height: {snapshot_block_height} - {datetime.datetime.now()}")

        ddf = load_and_prepare_data(snapshot_block_height)
        G = nx.read_graphml(join(VALIDATED_PROJECTIONS_INPUT_PATH, f'validated_token_projection_graph_{snapshot_block_height}.graphml'))
        all_cliques = get_cliques(G)

        for filter_method in ['upper_bound', 'lower_bound']:
            cliques_snapshot, cliques_snapshot_control, cliques_pvalues = analyze_cliques(snapshot_block_height, snapshot_date, ddf, all_cliques, filter_method)

            cliques[filter_method]['sample'][snapshot_date] = cliques_snapshot
            cliques[filter_method]['control'][snapshot_date] = cliques_snapshot_control
            cliques[filter_method]['pvalues'][snapshot_date] = cliques_pvalues

    return cliques

if __name__ == "__main__":
    cliques = cliques_main()
    output_path = join(path, 'data/cliques_data_class.pkl')

    with open(output_path, 'wb') as handle:
        pickle.dump(cliques, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Cliques data saved to {output_path}")
