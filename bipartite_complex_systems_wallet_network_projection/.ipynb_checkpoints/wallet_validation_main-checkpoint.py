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

# os 
import os
from os.path import join
from dotenv import load_dotenv

# auxillary 
from tqdm import tqdm

# data storage
import h5py

# custom imports 
from functions import hypergeom_wallets, process_batch
from data_loading.load_data import load_preparations, load_snapshot_frame



# load environment
load_dotenv()

# specify paths
path = os.environ["PROJECT_PATH"]

output_path = join(path, 'wallet_projection_output/output_f-none')

# load snapshot & 
snapshots, token_addresses = load_preparations()


pct_supply_coverage_list = {}

for snapshot in snapshots:
    print(f"Snapshot for Block Height: {snapshot}") 
    
    # Load data
    ddf = load_snapshot_frame(snapshot, path, 0.000005)
    
    pop_size = len(ddf.token_address.unique())

    # Save coverage
    pct_supply_coverage = dict(ddf.groupby('token_address').pct_supply.sum())
    
     # Save in dict 
    pct_supply_coverage_list[snapshot] = pct_supply_coverage

    # Create an HDF5 file for the current snapshot
    file_name = f'output_snapshot_{snapshot}.h5'
    
    with h5py.File(join(output_path, file_name), 'w') as file:
        
        # Create a group for the current snapshot
        snapshot_group = file.create_group(f'snapshot_{snapshot}')

        # Define the batch size for computing the p-values
        batch_size = 1024

        # Generate all possible combinations
        combinations_generator = combinations(ddf.address.unique(), 2)

        # Calculate the number of combinations
        num_combinations = math.comb(len(ddf.address.unique()), 2)

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
        



