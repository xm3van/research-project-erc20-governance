import random
import datetime
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ast import literal_eval
from os.path import join
import os
from dotenv import load_dotenv
from src.utilities.metrics_and_tests import *
from src.utilities.utils import *
from src.analysis.link_analysis import LinkAnalysis
import pickle
import multiprocessing

load_dotenv()

path = os.environ['DATA_DIRECTORY']
df_snapshots = pd.read_csv('data/snapshot_selection.csv')
df_tokens = pd.read_csv("data/final_token_selection.csv")
df_token_price = pd.read_csv("data/price_table.csv", index_col=0)

TOKEN_BALANCE_TABLE_INPUT_PATH = join(path, "data/snapshot_token_balance_tables_enriched")
VALIDATED_PROJECTIONS_INPUT_PATH = join(path, 'data/validated_token_projection_graphs')
START_BLOCK_HEIGHT = 11659570
SENSITIVITY_ANALYSIS = True #NOTE: False run with Reference value 0.000005 ~ 0.0005% of supply 

# remove burner addresses
known_burner_addresses = ['0x0000000000000000000000000000000000000000',
                          '0x0000000000000000000000000000000000000000',
                          '0x0000000000000000000000000000000000000001',
                          '0x0000000000000000000000000000000000000002',
                          '0x0000000000000000000000000000000000000003',
                          '0x0000000000000000000000000000000000000004',
                          '0x0000000000000000000000000000000000000005',
                          '0x0000000000000000000000000000000000000006',
                          '0x0000000000000000000000000000000000000007',
                          '0x000000000000000000000000000000000000dead']

def analyze_link(link, ddf, token_lookup):
    results = {}
    results_sample_population = {}
    pvalues = {}
    results_directional = {}
    results_sample_population_directional = {}
    pvalues_directional = {}

    analyzer = LinkAnalysis(link, ddf, None, None, token_lookup)
    analyzer.directional = False
    link_members_unique = analyzer.link_member_wallets()

    if not link_members_unique:
        return None

    filter1 = ddf.address.isin(link_members_unique)

    ddf_sample = ddf[filter1].copy()
    ddf_sample_population = ddf.copy()

    analyzer.sub_dataFrame = ddf_sample
    analyzer.sub_dataFrame_sample_population = ddf_sample_population
    link_name, res, res_sample_population, pvals = analyzer.analyze_link()

    results[str(link_name)] = res
    results_sample_population[str(link_name)] = res_sample_population
    pvalues[str(link_name)] = pvals

    for token in link:
        filter1 = ddf.token_address == token
        filter2 = ddf.address.isin(link_members_unique)

        ddf_sample_directional = ddf[filter2 & filter1].copy()
        token_name = token_lookup[token]

        ddf_sample_population_directional = ddf[filter1].copy()

        analyzer_directional = LinkAnalysis(link, ddf, ddf_sample_directional, ddf_sample_population_directional, token_lookup)
        analyzer_directional.directional = True
        link_name, res, res_sample_population, pvals = analyzer_directional.analyze_link()

        results_directional[f'{link_name}: {token_name}'] = res
        results_sample_population_directional[f'{link_name}: {token_name}'] = res_sample_population
        pvalues_directional[f'{link_name}: {token_name}'] = pvals

    return (results, results_sample_population, pvalues, results_directional, results_sample_population_directional, pvalues_directional)

def process_snapshot(snapshot_data, supply_threshold=0.000005):
    snapshot_date, snapshot_block_height, df_tokens, df_token_price = snapshot_data
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
    all_links = list(G.edges())
    all_links = [sorted(i) for i in all_links]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.starmap(analyze_link, [(link, ddf, token_lookup) for link in all_links])
    pool.close()
    pool.join()

    links_snapshot = {}
    links_snapshot_sample_population = {}
    links_pvalues = {}
    links_snapshot_directional = {}
    links_snapshot_sample_population_directional = {}
    links_pvalues_directional = {}

    for result in results:
        if result:
            res, res_sample_population, pvals, res_dir, res_sample_pop_dir, pvals_dir = result
            links_snapshot.update(res)
            links_snapshot_sample_population.update(res_sample_population)
            links_pvalues.update(pvals)
            links_snapshot_directional.update(res_dir)
            links_snapshot_sample_population_directional.update(res_sample_pop_dir)
            links_pvalues_directional.update(pvals_dir)

    return snapshot_date, links_snapshot, links_snapshot_sample_population, links_pvalues, links_snapshot_directional, links_snapshot_sample_population_directional, links_pvalues_directional

def links_main(supply_threshold=0.000005):
    links = {'sample': {}, 'sample_population': {}, 'pvalues': {}, 'sample_directional': {}, 'sample_population_directional': {}, 'pvalues_directional': {}}

    snapshots_data = [(row['Date'], row['Block Height'], df_tokens, df_token_price) for _, row in df_snapshots[df_snapshots['Block Height'] >= START_BLOCK_HEIGHT].iterrows()]

    for snapshot_data in snapshots_data:
        snapshot_date, links_snapshot, links_snapshot_sample_population, links_pvalues, links_snapshot_directional, links_snapshot_sample_population_directional, links_pvalues_directional = process_snapshot(snapshot_data, supply_threshold)

        links['sample'][snapshot_date] = links_snapshot
        links['sample_population'][snapshot_date] = links_snapshot_sample_population
        links['pvalues'][snapshot_date] = links_pvalues

        links['sample_directional'][snapshot_date] = links_snapshot_directional
        links['sample_population_directional'][snapshot_date] = links_snapshot_sample_population_directional
        links['pvalues_directional'][snapshot_date] = links_pvalues_directional

    return links

if __name__ == "__main__":

    if SENSITIVITY_ANALYSIS == False: 
        links = links_main()  # Store the returned links dictionary

        # Specify the path to save the links data
        output_path = join(path, 'output/links/metrics/links_data.pkl')

        # Serialize and save the links dictionary
        with open(output_path, 'wb') as handle:
            pickle.dump(links, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"links data saved to {output_path}")

    else:
        # search range 
        supply_thresholds = [0.05, 0.005, 0.0005, 0.00005, 0.000005, 0.0000005, 0.0000005]

        for supply_threshold in supply_thresholds: 

            print(f"Supply Threshold: {supply_threshold}")


            cliques = links_main(supply_threshold)
            output_path = join(path, f'output/links/metrics/links_data_{supply_threshold}.pkl')

            with open(output_path, 'wb') as handle:
                pickle.dump(cliques, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"links data saved to {output_path}")

