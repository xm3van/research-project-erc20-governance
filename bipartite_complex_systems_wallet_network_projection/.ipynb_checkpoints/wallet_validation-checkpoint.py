from scipy.stats import hypergeom 
import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures

import os

from os.path import join

from statsmodels.stats.multitest import multipletests as m_tests


#custom functions 
# from wallet_hypergeom import main_wallets

from dotenv import load_dotenv

load_dotenv()

path = os.environ["PROJECT_PATH"]


# snapshot selection
df_snapshot = pd.read_csv("../assets/snapshot_selection.csv")

# address selection
df_addresses = pd.read_csv("../assets/df_final_token_selection_20221209.csv")

# burner addresses
# remove burner addresses
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



output_path = '/local/scratch/exported/governance-erc20/project_erc20_governance_data/wallet_projection_output/output_f-none'


def main_wallets(wallet1, wallet2, pop_size):

    # unique token a given address holds
    wallet1_uniqT = ddf[ddf.address == wallet1].token_address.unique()
    wallet2_uniqT = ddf[ddf.address == wallet2].token_address.unique()

    # calcualte intersection in token holdings (binary)
    wallet1_wallet2_uniqT_intersection = np.intersect1d(wallet1_uniqT,wallet2_uniqT, assume_unique=True)

    # calculate hyptogeometric
    # Define the parameters of the distribution
    M = pop_size  # number of token at a given snapshot    
    n = len(wallet1_uniqT)  # number of draws - number of token wallets1
    K = len(wallet2_uniqT)  # number of successes in population - number of token wallets2
    x = len(wallet1_wallet2_uniqT_intersection) # number of successes in draws - intersection of tokens in wallet1 and wallet2

    # Compute the cumulative probability of obtaining at most x successes
    pvalue = 1 - hypergeom.cdf(x, M, n, K)
    
    # print(f'token_address {address1} has {len_token1} Unique Addresses | token_address {address2} has {len_token2} Unique Addresses | Intersection: {len_intersection}, p value: {pvalue}')

    return pvalue


# Define a function to compute the p-values for a batch of wallet address pairs
def compute_pvalues(wallet_pairs, pop_size):
    pvalues_batch = []
    for pair in wallet_pairs:
        if pair in processed_pairs:
            # Skip pairs that have already been processed
            continue
        else:
            # Compute the p-value for the current pair of wallet addresses
            pvalue = main_wallets(pair[0], pair[1], pop_size) ##CHANGE BACK TO NON-TEXT

            # Store the p-value in the batch list
            pvalues_batch.append(pvalue)

    # Return a tuple of the wallet address pairs and their corresponding p-values
    return wallet_pairs, pvalues_batch

### saves how much circulating supply on ethereum has been covered 
pct_supply_coverage = {} 

for snapshot in df_snapshot[df_snapshot['Block Height'] >= 11547458]['Block Height']:
    
    print(f"Snapshot for Block Height: {snapshot}") 

    ## formating of data
    # load data
    ddf = pd.read_csv(
        join(
            path,
            f"token_balance_lookup_tables_labelled/df_token_balenace_labelled_greater_01pct_bh{snapshot}_v2.csv"
        )
    )
    
    ## rename column
    ddf = ddf.rename(columns={'address_x': 'address'})

    # filter data
    ddf = ddf[ddf.value > 0]
    ddf = ddf[ddf.token_address.isin(df_addresses.address) == True]

    # remove known burner addresses
    ddf = ddf[ddf.address.isin(known_burner_addresses) == False]
    
    # we set a cut of point of 0.001% 
    ddf = ddf[ddf.pct_supply > 0.00001]  
    
    # save coverage 
    pct_supply_coverage[str(snapshot)] = dict(ddf.groupby('token_address').pct_supply.sum())
   
    # population is the size of the possible token that can be held in the sample 
    pop_size = len(ddf.token_address.unique())
    
    # Define the batch size for computing the p-values
    batch_size = 10_000

    # Compute a set of all unique pairs of wallet addresses that have already been processed
    processed_pairs = set()

    # temp storage
    pvalues_dict = {}


    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        num_pairs = len(ddf.address.unique())**2
        counter = 0
        threshold = num_pairs/20
        with tqdm(total=num_pairs, desc='Job Submission') as pbar_submit:
            for i in range(0, len(ddf.address.unique()), batch_size):
                # Get a batch of wallet addresses
                wallets_batch = ddf.address.unique()[i:i+batch_size]

                # Submit a job to compute the p-values for pairs of wallet addresses that have not yet been processed
                for j in range(i+batch_size, len(ddf.address.unique()), batch_size):
                    # Get another batch of wallet addresses
                    wallets_batch2 = ddf.address.unique()[j:j+batch_size]

                    # Compute the set of pairs of wallet addresses to process
                    wallet_pairs = [(w1, w2) for w1 in wallets_batch for w2 in wallets_batch2]

                    # Submit a job to compute the p-values for the current batch of wallet address pairs
                    futures.append(executor.submit(compute_pvalues, wallet_pairs, pop_size))

                    counter += len(wallet_pairs)
                    # Update the job submission progress bar
                    if counter >= threshold:
                        pbar_submit.update(threshold)
                        counter = 0 

        # Collect the results from the completed jobs
        counter = 0
        threshold = len(futures)/20
        with tqdm(total=len(futures), desc='Processing Jobs') as pbar_process:
            for future in concurrent.futures.as_completed(futures):
                # Get the results from the completed job
                result = future.result()
                if result is not None:
                    wallet_pairs, pvalues_batch = result

                    # Store the p-values in a dictionary keyed by the wallet address pairs
                    for i, pair in enumerate(wallet_pairs):
                        pvalues_dict[tuple(pair)] = pvalues_batch[i]

                        # Update the set of processed pairs
                        processed_pairs.update([tuple(pair)])


                # Update the processing progress bar
                counter += len(wallet_pairs)
                    
                # Update the job submission progress bar
                if counter >= threshold:
                    pbar_process.update(threshold)
                    counter = 0 
                


            
            
    # convert p_values dict to df
    
    ## Evaluate pvalues
    # store pvalues
    df_pvalues = pd.DataFrame.from_dict(pvalues_dict, orient="index")
    df_pvalues.reset_index(inplace=True)
    df_pvalues.columns = ["combination", "p_value"]

    # value test
    m_test = m_tests(pvals=df_pvalues.p_value, alpha=0.01, method="bonferroni")
    df_pvalues["m_test_result"] = m_test[0]
    df_pvalues["m_test_value"] = m_test[1]
    df_pvalues.to_csv(join(output_path, f'pvalues_wallets_{snapshot}.csv'), mode='a', header=not exists(join(output_path, 'pvalues_wallets.csv')))
