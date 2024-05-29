import random
import datetime
import pandas as pd
import numpy as np
import networkx as nx 

import matplotlib.pyplot as plt
from ast import literal_eval

from os.path import join
import os

# from token_projection.token_projection import *
from src.utilities.metrics_and_tests import *
from src.utilities.utils import * 

from src.analysis.link_analysis import * 

import pickle  # Add this import to the top of your script

from dotenv import load_dotenv
load_dotenv()  

path = os.environ['DATA_DIRECTORY']
df_snapshots = pd.read_csv('data/snapshot_selection.csv')
df_tokens = pd.read_csv("data/final_token_selection.csv")
df_token_price = pd.read_csv("data/price_table.csv", index_col=0)

TOKEN_BALANCE_TABLE_INPUT_PATH = join(path, "data/snapshot_token_balance_tables_enriched")
VALIDATED_PROJECTIONS_INPUT_PATH = join(path, 'data/validated_token_projection_graphs')

# df_tokens = pd.read_csv('../assets/df_final_token_selection_20230813.csv')

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

### NOTE: YOU NEED TO RE-RUN TOKEN VALIDATION --> REMOVAL OF BITDAO CAUSED THIS.
### ALSO DOUBLE CHECK THAT BITDAO IS THE PROBLEM AND NOT AURA !!! 



# Initialize a dictionary to store the links


def links_main(): 
    links = {'sample': {},'control': {}, 'pvalues': {}}

    for _, row in df_snapshots[df_snapshots['Block Height'] > 11547458].iterrows():

        snapshot_date = row['Date']
        snapshot_block_height = row['Block Height']

        print(f"Snapshot for Block Height: {snapshot_block_height} - {datetime.datetime.now()}")

        #### SNAPSHOT DATA
        # Load data
        ddf = pd.read_csv(join(TOKEN_BALANCE_TABLE_INPUT_PATH, f'token_holder_snapshot_balance_labelled_{snapshot_block_height}.csv'))

        # remove 
        ddf = ddf[~ddf.address.isin(known_burner_addresses)]

        # only include wallet holding more than 0.000005 ~ 0.0005% of supply 
        ddf = ddf[ddf.pct_supply > 0.000005] 

        # portfolio price 
        # ddf['contract_decimals'] = int # find a better solution 
        ddf['token_price_usd'] = float

        for contract_address in ddf.token_address.unique(): 

            token_price = df_token_price.loc[str(contract_address), str(snapshot_block_height)]

            ddf.loc[ddf.token_address == contract_address, 'token_price_usd'] = token_price


        ddf['value_usd'] = (ddf['value']/(10**18)*ddf['token_price_usd'])



        #### link DATA 
        token_lookup = df_tokens[['address','symbol']].set_index('address')['symbol'].to_dict()

        # Create an empty graph
        G = nx.read_graphml(join(VALIDATED_PROJECTIONS_INPUT_PATH, f'validated_token_projection_graph_{snapshot_block_height}.graphml'))
        nx.set_node_attributes(G, token_lookup, 'name')

        # Find all links in the graph
        all_links = list(G.edges())

        # Ensure that links are always ordered the same 
        # Prevents maker-yearn and yearn-maker being counted separately
        all_links = [sorted(i) for i in all_links]


        # list all links of the given snapshot for analysis  
     

        links_snapshot = {}
        links_snapshot_control = {} 
        links_pvalues = {}


        for link in all_links:
            
            link_members_unique = link_member_wallets(link, ddf)


            if link_members_unique == []: 

                # links_snapshot[str(link_name)] = {}
                # links_snapshot_control[str(link_name)] = {}
                # links_pvalues[str(link_name)] = {}

                print("Skip")

                pass


            else:

                ## Sample DataFrame
                ddf_sub = ddf[ddf.address.isin(link_members_unique)].copy()  

                ### tokenholder population of the link
                    # filter 1: token holder
                filter1 = ddf.token_address.isin(link)
                    # filter 2: Independence for t-test (control cannot have overlap with sample) 
                filter2 = ddf['address'].isin(link_members_unique) 

                relevant_population =  list(ddf[(filter1==True) & (filter2!=True)].address.unique()) 

                ## control dataframe 
                # random.seed(42) # reproducibility
                control_sample = random.sample(relevant_population, len(link_members_unique))
                ddf_sub_control = ddf[ddf.address.isin(control_sample)].copy()  


                ### ANALYSIS 
                link_name, results, results_control, pvalues = analyse_link(ddf_sub, ddf_sub_control, ddf, link, token_lookup)


                links_snapshot[str(link_name)] = results
                links_snapshot_control[str(link_name)] = results_control
                links_pvalues[str(link_name)] = pvalues

        # PROBLEM: FIX HOW TO STORE VALUES PER link PER SNAPSHOT
        links['sample'][snapshot_date] = links_snapshot
        links['control'][snapshot_date] = links_snapshot_control
        links['pvalues'][snapshot_date] = links_pvalues
        
        
    return links 


if __name__ == "__main__":
    links = links_main()  # Store the returned links dictionary

    # Specify the path to save the links data
    output_path = join(path, 'data/links_data.pkl')  # Ensure `path` is correctly defined in your environment
        # Serialize and save the links dictionary
    with open(output_path, 'wb') as handle:
        pickle.dump(links, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"links data saved to {output_path}")


