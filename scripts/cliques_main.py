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
import multiprocessing

from src.utilities.metrics_and_tests import *
from src.utilities.utils import *
from src.analysis.clique_analysis import CliqueAnalysis

load_dotenv()

# Load environment variables
path = os.environ['DATA_DIRECTORY']
TOKEN_BALANCE_TABLE_INPUT_PATH = join(path, "data/snapshot_token_balance_tables_enriched")
VALIDATED_PROJECTIONS_INPUT_PATH = join(path, 'data/validated_token_projection_graphs')
START_BLOCK_HEIGHT = 11659570
SENSITIVITY_ANALYSIS = True #NOTE: False run with Reference value 0.000005 ~ 0.0005% of supply 


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

def analyze_clique(clique, ddf, token_lookup, filter_method):
    results = {}
    results_control = {}
    pvalues = {}
    results_directional = {}
    results_control_directional = {}
    pvalues_directional = {}

    analyzer = CliqueAnalysis(clique, ddf, None, None, token_lookup)
    analyzer.directional = False
    
    if filter_method == 'weak_estimate':
        clique_members_unique = analyzer.clique_member_wallets_weak()
    else:
        clique_members_unique = analyzer.clique_member_wallets_strong()

    if not clique_members_unique:
        return None

    ddf_sub = ddf[ddf.address.isin(clique_members_unique)].copy()
    ddf_sub_control = ddf.copy()

    analyzer.sub_dataFrame = ddf_sub
    analyzer.sub_dataFrame_sample_population = ddf_sub_control

    clique_name, res, res_control, pvals = analyzer.analyze_clique()

    results[str(clique_name)] = res
    results_control[str(clique_name)] = res_control
    pvalues[str(clique_name)] = pvals

    for token in clique:
        token_name = token_lookup[token]
        
        filter1 = ddf.token_address == token
        filter2 = ddf.address.isin(clique_members_unique)
        
        ddf_sub_directional = ddf[filter1 & filter2].copy()
        ddf_control_directional = ddf[filter1].copy()

        if len(ddf_control_directional) < len(ddf_sub_directional):
            print(f"[WARNING]: Skip directional analysis for {token_name} due to insufficient control population")
            continue

        analyzer_directional = CliqueAnalysis(clique, ddf, ddf_sub_directional, ddf_control_directional, token_lookup)
        analyzer_directional.directional = True
        clique_name, res, res_control, pvals = analyzer_directional.analyze_clique()

        results_directional[f'{clique_name}: {token_name}'] = res
        results_control_directional[f'{clique_name}: {token_name}'] = res_control
        pvalues_directional[f'{clique_name}: {token_name}'] = pvals

    return (results, results_control, pvalues, results_directional, results_control_directional, pvalues_directional)

def process_snapshot(snapshot_data, supply_threshold=0.000005):
    snapshot_date, snapshot_block_height, df_tokens, df_token_price, filter_method = snapshot_data
    print(f"Snapshot for Block Height: {snapshot_block_height} - {datetime.datetime.now()}")

    ddf = pd.read_csv(join(TOKEN_BALANCE_TABLE_INPUT_PATH, f'token_holder_snapshot_balance_labelled_{snapshot_block_height}.csv'))
    ddf = ddf[ddf.token_address.str.lower().isin(df_tokens.address.str.lower())]
    ddf = ddf[~ddf.address.isin(known_burner_addresses)]
    ddf = ddf[ddf.pct_supply >= supply_threshold]
    ddf['token_price_usd'] = ddf['token_address'].apply(lambda x: df_token_price.loc[str(x), str(snapshot_block_height)])
    ddf['value_usd'] = ddf['value'] / (10**18) * ddf['token_price_usd']
    token_lookup = df_tokens[['address', 'symbol']].set_index('address')['symbol'].to_dict()
    G = nx.read_graphml(join(VALIDATED_PROJECTIONS_INPUT_PATH, f'validated_token_projection_graph_{snapshot_block_height}.graphml'))
    nx.set_node_attributes(G, token_lookup, 'name')
    all_cliques = [sorted(clique) for clique in nx.find_cliques(G) if len(clique) > 2]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.starmap(analyze_clique, [(clique, ddf, token_lookup, filter_method) for clique in all_cliques])
    pool.close()
    pool.join()

    cliques_snapshot = {}
    cliques_snapshot_control = {}
    cliques_pvalues = {}
    cliques_snapshot_directional = {}
    cliques_snapshot_control_directional = {}
    cliques_pvalues_directional = {}

    for result in results:
        if result:
            res, res_control, pvals, res_dir, res_control_dir, pvals_dir = result
            cliques_snapshot.update(res)
            cliques_snapshot_control.update(res_control)
            cliques_pvalues.update(pvals)
            cliques_snapshot_directional.update(res_dir)
            cliques_snapshot_control_directional.update(res_control_dir)
            cliques_pvalues_directional.update(pvals_dir)

    return snapshot_date, filter_method, cliques_snapshot, cliques_snapshot_control, cliques_pvalues, cliques_snapshot_directional, cliques_snapshot_control_directional, cliques_pvalues_directional

def cliques_main(supply_threshold=0.000005):
    cliques = {
        'weak_estimate': {'sample': {}, 'sample_population': {}, 'pvalues': {}, 'sample_directional': {}, 'sample_population_directional': {}, 'pvalues_directional': {}},
        'strong_estimate': {'sample': {}, 'sample_population': {}, 'pvalues': {}, 'sample_directional': {}, 'sample_population_directional': {}, 'pvalues_directional': {}}
    }

    snapshots_data = [(row['Date'], row['Block Height'], df_tokens, df_token_price, filter_method) for _, row in df_snapshots[df_snapshots['Block Height'] >= START_BLOCK_HEIGHT].iterrows() for filter_method in ['weak_estimate', 'strong_estimate']]

    for snapshot_data in snapshots_data:
        snapshot_date, filter_method, cliques_snapshot, cliques_snapshot_control, cliques_pvalues, cliques_snapshot_directional, cliques_snapshot_control_directional, cliques_pvalues_directional = process_snapshot(snapshot_data, supply_threshold=supply_threshold)

        cliques[filter_method]['sample'][snapshot_date] = cliques_snapshot
        cliques[filter_method]['sample_population'][snapshot_date] = cliques_snapshot_control
        cliques[filter_method]['pvalues'][snapshot_date] = cliques_pvalues

        cliques[filter_method]['sample_directional'][snapshot_date] = cliques_snapshot_directional
        cliques[filter_method]['sample_population_directional'][snapshot_date] = cliques_snapshot_control_directional
        cliques[filter_method]['pvalues_directional'][snapshot_date] = cliques_pvalues_directional

    return cliques


if __name__ == "__main__":

    if SENSITIVITY_ANALYSIS == False: 

        cliques = cliques_main()
        output_path = join(path, 'output/cliques/metrics/cliques_data.pkl')

        with open(output_path, 'wb') as handle:
            pickle.dump(cliques, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Cliques data saved to {output_path}")

    else:
        # search range 
        # supply_thresholds = [0.05, 0.005, 0.0005, 0.00005, 0.000005, 0.0000005, 0.00000005]
        supply_thresholds = [0.00000005]

        for supply_threshold in supply_thresholds: 
            
            print(f"Supply Threshold: {supply_threshold}")

            cliques = cliques_main(supply_threshold)
            output_path = join(path, f'output/cliques/metrics/cliques_data_{supply_threshold}.pkl')

            with open(output_path, 'wb') as handle:
                pickle.dump(cliques, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Cliques data saved to {output_path}")

