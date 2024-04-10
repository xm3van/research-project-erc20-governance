# imports
import pandas as pd
import numpy as np
from scipy.stats import hypergeom 
from statsmodels.stats.multitest import multipletests
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

import os
from os.path import join

from src.token_projection.token_projection_analysis import * 


from dotenv import load_dotenv
load_dotenv()  

path = os.environ['DATA_DIRECTORY']


# Constants
SNAPSHOT_CSV_PATH = 'data/snapshot_selection.csv'
ADDRESS_CSV_PATH = 'data/final_token_selection.csv'
OUTPUT_PATH = join(path, 'data/validated_token_projection_graphs')

KNOWN_BURNER_ADDRESSES = [
    '0x0000000000000000000000000000000000000000', '0x000000000000000000000000000000000000dead',
    '0x0000000000000000000000000000000000000001', '0x0000000000000000000000000000000000000002',
    '0x0000000000000000000000000000000000000003', '0x0000000000000000000000000000000000000004',
    '0x0000000000000000000000000000000000000005', '0x0000000000000000000000000000000000000006',
    '0x0000000000000000000000000000000000000007'
]




# Main function to load, process data, and generate network graphs
def generate_network_graphs():
    df_snapshot = pd.read_csv(SNAPSHOT_CSV_PATH)
    df_addresses = pd.read_csv(ADDRESS_CSV_PATH)
    for snapshot in df_snapshot[df_snapshot['Block Height']>=11659570]['Block Height']:
        ddf = pd.read_csv(join(path, f'data/snapshot_token_balance_tables_enriched/token_holder_snapshot_balance_labelled_{snapshot}.csv'))
        ddf = ddf[ddf['value'] > 0]
        ddf = ddf[~ddf['address'].isin(KNOWN_BURNER_ADDRESSES)]
        ddf = ddf[ddf['token_address'].isin(df_addresses['address'])]
        
        present_addresses = ddf['token_address'].unique()
        pop_size = len(ddf['address'].unique())
        
        validated_links = validate_token_links(present_addresses, ddf, pop_size)
        
        G = nx.Graph()
        G.add_edges_from(validated_links['combination'])
        
        # Store the graph with a name based on snapshot height
        nx.write_graphml(G, join(OUTPUT_PATH, f'validated_token_projection_graph_{snapshot}.graphml'))
        
        print(f"Generated and saved validated_token_projection_graph_{snapshot}")



if __name__ == "__main__":
    generate_network_graphs()