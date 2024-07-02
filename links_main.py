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

load_dotenv()  

path = os.environ['DATA_DIRECTORY']
df_snapshots = pd.read_csv('data/snapshot_selection.csv')
df_tokens = pd.read_csv("data/final_token_selection.csv")
df_token_price = pd.read_csv("data/price_table.csv", index_col=0)

TOKEN_BALANCE_TABLE_INPUT_PATH = join(path, "data/snapshot_token_balance_tables_enriched")
VALIDATED_PROJECTIONS_INPUT_PATH = join(path, 'data/validated_token_projection_graphs')
START_BLOCK_HEIGHT = 11659570
SUPPLY_THRESHOLD = 0 # given as faction (i.e. 1% = 0.01)

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

    links = {'sample': {}, 'sample_population': {}, 'pvalues': {}, 'sample_directional':{}, 'sample_population_directional':{}, 'pvalues_directional':{}}


    for _, row in df_snapshots[df_snapshots['Block Height'] >= START_BLOCK_HEIGHT].iterrows():

        snapshot_date = row['Date']
        snapshot_block_height = row['Block Height']

        print(f"Snapshot for Block Height: {snapshot_block_height} - {datetime.datetime.now()}")

        # Load data
        ddf = pd.read_csv(join(TOKEN_BALANCE_TABLE_INPUT_PATH, f'token_holder_snapshot_balance_labelled_{snapshot_block_height}.csv'))
        
        # Ensure we only check for tokens we want to analyze
        ddf = ddf[ddf.token_address.str.lower().isin(df_tokens.address.str.lower())]

        # Remove burner addresses
        ddf = ddf[~ddf.address.isin(known_burner_addresses)]

        # Only include wallets holding more than x% of supply 
        ddf = ddf[ddf.pct_supply > SUPPLY_THRESHOLD] 

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
        links_snapshot_sample_population = {} 
        links_pvalues = {}
        
        links_snapshot_directional = {}
        links_snapshot_sample_population_directional= {} 
        links_pvalues_directional = {}

        # START PARALLELIZE 
        for link in all_links:
            analyzer = LinkAnalysis(link, ddf, None, None, token_lookup)
            analyzer.directional = False 
            link_members_unique = analyzer.link_member_wallets()

            if not link_members_unique:
                print("Skip")
                continue

            # filter 
            filter1 = ddf.address.isin(link_members_unique)

            # Sample DataFrame
            ddf_sample = ddf[filter1].copy()  
            
            # Population 
            ddf_sample_population = ddf.copy()
            
            # Perform analysis
            analyzer.sub_dataFrame = ddf_sample
            analyzer.sub_dataFrame_sample_population = ddf_sample_population
            link_name, results, results_sample_population, pvalues = analyzer.analyze_link()

            links_snapshot[str(link_name)] = results
            links_snapshot_sample_population[str(link_name)] = results_sample_population
            links_pvalues[str(link_name)] = pvalues
            
            
            # Directional            
            for token in link: 

                # filter 
                filter1 = ddf.token_address == token
                filter2 = ddf.address.isin(link_members_unique)
                
                # Sample DataFrame - contains clique members but we only look at one token of a link
                ddf_sample_directional = ddf[filter2 & filter1].copy() 
                
                # For naming convention
                token_name = token_lookup[token]
                
                # Sample sample_population Population are token members which are not part of identified link members 
                ddf_sample_population_directional = ddf[filter1].copy() 
                
                # update analzyer to directional 
                analyzer_directional = LinkAnalysis(link, ddf,  ddf_sample_directional, ddf_sample_population_directional, token_lookup)

                # set directional to False 
                analyzer_directional.directional = True
                link_name, results, results_sample_population, pvalues = analyzer_directional.analyze_link()
                
                links_snapshot_directional[f'{link_name}: {token_name}'] = results
                links_snapshot_sample_population_directional[f'{link_name}: {token_name}'] = results_sample_population
                links_pvalues_directional[f'{link_name}: {token_name}'] = pvalues


        # END PARALLELIZE 

        links['sample'][snapshot_date] = links_snapshot
        links['sample_population'][snapshot_date] = links_snapshot_sample_population
        links['pvalues'][snapshot_date] = links_pvalues
        
        links['sample_directional'][snapshot_date] = links_snapshot_directional
        links['sample_population_directional'][snapshot_date] = links_snapshot_sample_population_directional
        links['pvalues_directional'][snapshot_date] = links_pvalues_directional

    return links 

if __name__ == "__main__":
    links = links_main()  # Store the returned links dictionary

    # Specify the path to save the links data
    output_path = join(path, 'data/links_data.pkl') 
    
    # Serialize and save the links dictionary
    with open(output_path, 'wb') as handle:
        pickle.dump(links, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"links data saved to {output_path}")
