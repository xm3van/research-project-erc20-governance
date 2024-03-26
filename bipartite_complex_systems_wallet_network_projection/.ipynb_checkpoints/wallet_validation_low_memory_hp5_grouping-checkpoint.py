### .p5 file implementation

# data  packages
import pandas as pd
import numpy as np

# data manipulation packages
import math
from scipy.stats import hypergeom 
from itertools import combinations, islice

# data processing 
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# 
from data_processing.token_price import convert_blockheight_to_date, usd_value_decimals_token

# os 
import os
from os.path import join
from dotenv import load_dotenv

# auxillary 
from tqdm import tqdm

# data storage
import h5py

# timestamping 
import datetime


# load environment
load_dotenv()

# specify paths
path = os.environ["PROJECT_PATH"]
covalent_api_key =  os.environ["COVALENTHQ_API_KEY"]
output_path = '/local/scratch/exported/governance-erc20/project_erc20_governance_data/wallet_projection_output/output_f-none_grouping'

# Snapshot selection
df_snapshot = pd.read_csv("../assets/snapshot_selection.csv")

# Address selection
df_addresses = pd.read_csv("../assets/df_final_token_selection_20230813.csv")

# Burner addresses (remove burner addresses)
known_burner_addresses = [
    "0x0000000000000000000000000000000000000000",
    "0x0000000000000000000000000000000000000000",
    "0x0000000000000000000000000000000000000001",
    "0x0000000000000000000000000000000000000002",
    "0x0000000000000000000000000000000000000003",
    "0x0000000000000000000000000000000000000004",
    "0x0000000000000000000000000000000000000005",
    "0x0000000000000000000000000000000000000006",
    "0x0000000000000000000000000000000000000007",
    "0x000000000000000000000000000000000000dead",
]



def main_wallets(wallet1, wallet2, pop_size, ddf):
    # Unique tokens a given address holds
    wallet1_uniqT = ddf[ddf.address == wallet1].token_address.unique()
    wallet2_uniqT = ddf[ddf.address == wallet2].token_address.unique()

    # Calculate intersection in token holdings (binary)
    wallet1_wallet2_uniqT_intersection = np.intersect1d(wallet1_uniqT, wallet2_uniqT, assume_unique=True)

    # Calculate hypergeometric
    M = pop_size  # Number of tokens at a given snapshot    
    n = len(wallet1_uniqT)  # Number of draws - Number of tokens in wallet1
    K = len(wallet2_uniqT)  # Number of successes in population - Number of tokens in wallet2
    x = len(wallet1_wallet2_uniqT_intersection)  # Number of successes in draws - Intersection of tokens in wallet1 and wallet2

    # Compute the cumulative probability of obtaining at most x successes
    pvalue = 1 - hypergeom.cdf(x, M, n, K)

    return np.array((wallet1, wallet2, pvalue))

# Define the function that will process a batch of combinations
def process_batch(batch):
    results = np.empty((len(batch), 3), dtype=object)
    
    for i, combination in enumerate(batch):
        # Call your function here to process the combination
        result = main_wallets(combination[0], combination[1], pop_size, ddf)
        results[i] = result
    return results


pct_supply_coverage_list = {}

for snapshot in df_snapshot[df_snapshot['Block Height'] >= 12244515]['Block Height']: # 11547458
    
    
    print(f"Snapshot for Block Height: {snapshot} - {datetime.datetime.now()}")
    
    # Load data
    ddf = pd.read_csv(join(path, f"token_balance_lookup_tables_labelled/df_token_balenace_labelled_greater_01pct_bh{snapshot}_v2.csv"))
    ddf = ddf.rename(columns={'address_x': 'address'})
    ddf = ddf[ddf.value > 0]
    ddf = ddf[ddf.token_address.isin(df_addresses.address)]
    ddf = ddf[~ddf.address.isin(known_burner_addresses)]
    
    # supply 
    dict_ts = dict(ddf.groupby('token_address').value.sum())
    ddf['pct_supply'] = 0
    for t in dict_ts.keys(): 
        ddf.loc[ddf.token_address == t, 'pct_supply'] = ddf[ddf.token_address == t].value / dict_ts[t]
    ddf = ddf[ddf.pct_supply > 0.000005] 
    
    # portfolio price 
    ddf['contract_decimals'] = int
    ddf['token_price_usd'] = float

    for contract_address in ddf.token_address.unique(): 

        response = usd_value_decimals_token(snapshot, contract_address, covalent_api_key)

        ddf.loc[ddf.token_address == contract_address, 'contract_decimals'] = response['contract_decimals']
        ddf.loc[ddf.token_address == contract_address, 'token_price_usd'] = response['token_price_usd']


    ddf['value_usd'] = (ddf['value']/(10**ddf['contract_decimals']))*ddf['token_price_usd']
    
            
    # grouping into log 0-90, 90-99, 99+ log value 
    ddf_grouping = pd.DataFrame(ddf.groupby('address').value_usd.sum()).reset_index()
    
    # Step 2: Calculate log-transformed portfolio values
    ddf_grouping['log_portfolio_value'] = ddf_grouping.value_usd.apply( lambda x: np.log10(x) ) 

    # Step 3: Determine percentiles for log-transformed values
    percentiles = [0, 90, 99, 100]
    percentile_values = np.percentile(ddf_grouping['log_portfolio_value'], percentiles)
    
    ## Step 3: Assign group labels based on the percentile ranges
    def assign_group(value):
        if value <= percentile_values[1]:
            return '0-90'
        elif percentile_values[1] < value <= percentile_values[2]:
            return '90-99'
        elif percentile_values[2] < value:
            return '99-100'

    ddf_grouping['group'] = ddf_grouping['log_portfolio_value'].apply(assign_group)



    # Contine with rest of script
    
    pop_size = len(ddf.token_address.unique())

    # Save coverage
    pct_supply_coverage = dict(ddf.groupby('token_address').pct_supply.sum())
    
     # Save in dict 
    pct_supply_coverage_list[snapshot] = pct_supply_coverage
    
    for group in ddf_grouping['group'].unique(): 
        
        # timestamp
        print(f"> Group: {group} - {datetime.datetime.now()}") 


        # Create an HDF5 file for the current snapshot
        file_name = f'output_snapshot_{snapshot}_{group}.h5'

        with h5py.File(join(output_path, file_name), 'w') as file:

            # Create a group for the current snapshot
            snapshot_group = file.create_group(f'snapshot_{snapshot}')

            # Define the batch size for computing the p-values
            batch_size = 1024

            # Generate all possible combinations
            combinations_generator = combinations(ddf_grouping[ddf_grouping.group == group].address.unique(), 2)

            # Calculate the number of combinations
            num_combinations = math.comb(len(ddf_grouping[ddf_grouping.group == group].address.unique()), 2)

            # Divide the combinations into batches
            batches = (list(islice(combinations_generator, batch_size)) for i in range(0, num_combinations, batch_size))

            # Create a group for storing the p-values
            p_values_group = snapshot_group.create_group('p_values')

            # Open the output file for writing
            with ProcessPoolExecutor(max_workers=12) as executor:
                results = []
                # Process each batch in parallel
                for batch in batches:
                    batch_result = executor.submit(process_batch, batch)
                    results.append(batch_result)

                # Wait for all results to be returned
                for i, future in enumerate(results):
                    batch_result = future.result()
                    # Write the batch result to the HDF5 file
                    p_values_group.create_dataset(f'batch_{i}', data=batch_result) 
                

df = pd.DataFrame(pct_supply_coverage_list)
df.to_csv(join(output_path, "pct_supply_coverage.csv"))
        



