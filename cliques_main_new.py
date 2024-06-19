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
from src.analysis.clique_analysis import CliqueAnalysis, clique_member_wallets_weak, clique_member_wallets_strong

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


def cliques_main():

    cliques = {
        'upper_bound': {'sample': {},'control': {},'pvalues': {}, 'sample_directional':{}, 'control_directional':{}, 'pvalues_directional':{}}, # change to adjust nomencalture to we upper to weak 
        'lower_bound': {'sample': {},'control': {},'pvalues': {}, 'sample_directional':{}, 'control_directional':{}, 'pvalues_directional':{}} # change to adjust nomencalture to we lower to strong 
    }
    
    insufficient_control_population={}
    
    for _, row in df_snapshots[df_snapshots['Block Height'] > 11547458].iterrows():
        snapshot_date = row['Date']
        snapshot_block_height = row['Block Height']

        print(f"Snapshot for Block Height: {snapshot_block_height} - {datetime.datetime.now()}")

        # Load and prepare data
        ddf = pd.read_csv(join(TOKEN_BALANCE_TABLE_INPUT_PATH, f'token_holder_snapshot_balance_labelled_{snapshot_block_height}.csv'))
        
        # Ensure we only check for tokens we want to analyze
        ddf = ddf[ddf.token_address.str.lower().isin(df_tokens.address.str.lower())]
        
        # Remove burner addresses
        ddf = ddf[~ddf.address.isin(known_burner_addresses)]
        
        # Only include wallets holding more than 0.000005 ~ 0.0005% of supply 
        ddf = ddf[ddf.pct_supply > 0.000005]
        
        # Assign token prices 
        ddf['token_price_usd'] = ddf['token_address'].apply(lambda x: df_token_price.loc[str(x), str(snapshot_block_height)])
        ddf['value_usd'] = ddf['value'] / (10 ** 18) * ddf['token_price_usd']

        # Get cliques
        G = nx.read_graphml(join(VALIDATED_PROJECTIONS_INPUT_PATH, f'validated_token_projection_graph_{snapshot_block_height}.graphml'))
        
        # Token lookup dictionary
        token_lookup = df_tokens[['address', 'symbol']].set_index('address')['symbol'].to_dict()
        nx.set_node_attributes(G, token_lookup, 'name')
        
        # filter out links due to node based clustering algo Brom 1973
        all_cliques = [sorted(clique) for clique in nx.find_cliques(G) if len(clique) > 2]

        # Analyze cliques
        for filter_method in ['upper_bound', 'lower_bound']:
            cliques_snapshot = {}
            cliques_snapshot_control = {}
            cliques_pvalues = {}
            
            cliques_snapshot_directional = {}
            cliques_snapshot_control_directional = {}
            cliques_pvalues_directional = {}

            for clique in all_cliques:
                if filter_method == 'upper_bound':
                    clique_members_unique = clique_member_wallets_weak(clique, ddf)
                else:
                    clique_members_unique = clique_member_wallets_strong(clique, ddf)

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
                
                
                for token in clique: 
                    
                    # For naming convention
                    token_name = token_lookup[token]
                    
                    # Control filters 
                    filter1 = ddf.token_address == token
                    filter2 = ddf.address.isin(clique_members_unique)
                    
                    # Sample DataFrame - contains clique members but we only look at one token of a clique
                    ddf_sub_directional = ddf[filter1 & filter2].copy() 

                    # Control Population are token members which are not part of identified clique members 
                    relevant_population_directional = ddf[filter1 & ~filter2].address.unique()
                    
                    clique_name = []
                    for t in clique: 
                        clique_name.append(token_lookup[t])
                    
                    
                    
                    
                    # check directional analysis sufficiency 
                    if len(relevant_population_directional) < len(clique_members_unique):
                        
                        if token_name in list(insufficient_control_population.keys()):
                            
                            insufficient_control_population[token_name].append(snapshot_block_height)
                            
                        else: 
                            insufficient_control_population[token_name] = [snapshot_block_height]
                            # log 
                            
                        log[f"{snapshot_block_height}-{snapshot_date}||{clique_name}:{token_name}"] = {
                        "population_size_directional": len(relevant_population_directional),
                        "sample_size_directional": len(clique_members_unique),
                        "token_population": len(ddf[filter1]), 
                        "filter_method": filter_method
                        }
                        
                        print(f"!!!! Skip directional analysis for {token_name} due to insufficient control population !!!!")
                        continue
                                              
                    control_directional = random.sample(list(relevant_population_directional), len(clique_members_unique))
                    ddf_control_directional = ddf[ddf.address.isin(control_directional)].copy()

                    # update analzyer to directional 
                    analyzer_directional = CliqueAnalysis(ddf_sub_directional, ddf_control_directional, ddf, clique, token_lookup)
                    clique_name, results, results_control, pvalues = analyzer_directional.analyze()

                    cliques_snapshot_directional[f'{clique_name}: {token_name}'] = results
                    cliques_snapshot_control_directional[f'{clique_name}: {token_name}'] = results_control
                    cliques_pvalues_directional[f'{clique_name}: {token_name}'] = pvalues
                    

            cliques[filter_method]['sample'][snapshot_date] = cliques_snapshot
            cliques[filter_method]['control'][snapshot_date] = cliques_snapshot_control
            cliques[filter_method]['pvalues'][snapshot_date] = cliques_pvalues
            
            cliques[filter_method]['sample_directional'][snapshot_date] = cliques_snapshot_directional
            cliques[filter_method]['control_directional'][snapshot_date] = cliques_snapshot_control_directional
            cliques[filter_method]['pvalues_directional'][snapshot_date] = cliques_pvalues_directional

    return cliques, insufficient_control_population, log

if __name__ == "__main__":
    cliques, insufficient_control_population, log = cliques_main()
    output_path = join(path, 'data/cliques_data_class.pkl')

    with open(output_path, 'wb') as handle:
        pickle.dump(cliques, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Cliques data saved to {output_path}")
    print(f"insufficient_control_population_tokens: {insufficient_control_population}")
    


