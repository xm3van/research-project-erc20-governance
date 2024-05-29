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
from src.analysis.link_analysis_class import LinkAnalysis

import pickle  # Add this import to the top of your script

load_dotenv()  

path = os.environ['DATA_DIRECTORY']
df_snapshots = pd.read_csv('data/snapshot_selection.csv')
df_tokens = pd.read_csv("data/final_token_selection.csv")
df_token_price = pd.read_csv("data/price_table.csv", index_col=0)

TOKEN_BALANCE_TABLE_INPUT_PATH = join(path, "data/snapshot_token_balance_tables_enriched")
VALIDATED_PROJECTIONS_INPUT_PATH = join(path, 'data/validated_token_projection_graphs')

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

def links_main(): 
    links = {'sample': {}, 'control': {}, 'pvalues': {}}

    for _, row in df_snapshots[df_snapshots['Block Height'] > 11547458].iterrows():

        snapshot_date = row['Date']
        snapshot_block_height = row['Block Height']

        print(f"Snapshot for Block Height: {snapshot_block_height} - {datetime.datetime.now()}")

        # Load data
        ddf = pd.read_csv(join(TOKEN_BALANCE_TABLE_INPUT_PATH, f'token_holder_snapshot_balance_labelled_{snapshot_block_height}.csv'))

        # Remove burner addresses
        ddf = ddf[~ddf.address.isin(known_burner_addresses)]

        # Only include wallets holding more than 0.000005 ~ 0.0005% of supply 
        ddf = ddf[ddf.pct_supply > 0.000005] 

        # Assign token prices
        ddf['token_price_usd'] = ddf['token_address'].apply(lambda x: df_token_price.loc[str(x), str(snapshot_block_height)])
        ddf['value_usd'] = ddf['value'] / (10**18) * ddf['token_price_usd']

        # Token lookup dictionary
        token_lookup = df_tokens[['address', 'symbol']].set_index('address')['symbol'].to_dict()

        # Create an empty graph
        G = nx.read_graphml(join(VALIDATED_PROJECTIONS_INPUT_PATH, f'validated_token_projection_graph_{snapshot_block_height}.graphml'))
        nx.set_node_attributes(G, token_lookup, 'name')

        # Find all links in the graph
        all_links = list(G.edges())
        all_links = [sorted(i) for i in all_links]

        links_snapshot = {}
        links_snapshot_control = {} 
        links_pvalues = {}

        for link in all_links:
            analyzer = LinkAnalysis(link, ddf, None, None, token_lookup)
            link_members_unique = analyzer.link_member_wallets()

            if not link_members_unique:
                print("Skip")
                continue

            # Sample DataFrame
            ddf_sub = ddf[ddf.address.isin(link_members_unique)].copy()  

            # Control DataFrame
            filter1 = ddf.token_address.isin(link)
            filter2 = ddf.address.isin(link_members_unique)
            relevant_population = ddf[filter1 & ~filter2].address.unique()
            control_sample = random.sample(list(relevant_population), len(link_members_unique))
            ddf_sub_control = ddf[ddf.address.isin(control_sample)].copy()

            # Perform analysis
            analyzer.sub_dataFrame = ddf_sub
            analyzer.sub_dataFrame_control = ddf_sub_control
            link_name, results, results_control, pvalues = analyzer.analyze_link()

            links_snapshot[str(link_name)] = results
            links_snapshot_control[str(link_name)] = results_control
            links_pvalues[str(link_name)] = pvalues

        links['sample'][snapshot_date] = links_snapshot
        links['control'][snapshot_date] = links_snapshot_control
        links['pvalues'][snapshot_date] = links_pvalues
        
    return links 

if __name__ == "__main__":
    links = links_main()  # Store the returned links dictionary

    # Specify the path to save the links data
    output_path = join(path, 'data/links_data_class.pkl')  # Ensure `path` is correctly defined in your environment
    # Serialize and save the links dictionary
    with open(output_path, 'wb') as handle:
        pickle.dump(links, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"links data saved to {output_path}")
