import pandas as pd
import numpy as np
from scipy.stats import hypergeom 
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def calculate_pvalue(address_pair, ddf, pop_size):
    address1, address2 = address_pair
    wallet1_uniqt = ddf[ddf.address == address1].token_address.unique()
    wallet2_uniqt = ddf[ddf.address == address2].token_address.unique()
    intersection = np.intersect1d(wallet1_uniqt, wallet2_uniqt, assume_unique=True)
    pvalue = 1 - hypergeom.cdf(len(intersection), pop_size, len(wallet1_uniqt), len(wallet2_uniqt))
    return (address_pair, pvalue)

def validate_wallet_links(present_addresses, ddf, pop_size):
    if len(present_addresses) <= 1:
        return {'nodes': np.nan, 'possible_nodes': len(present_addresses)}

    address_pairs = list(combinations(present_addresses, 2))
    p_dict = {}

    with ThreadPoolExecutor() as executor, tqdm(total=len(address_pairs), desc="Calculating p-values") as pbar:
        futures = {executor.submit(calculate_pvalue, pair, ddf, pop_size): pair for pair in address_pairs}
        
        for future in as_completed(futures):
            address_pair, pvalue = future.result()
            p_dict[address_pair] = pvalue
            pbar.update(1)

    df_pvalues = pd.DataFrame(list(p_dict.items()), columns=['combination', 'p_value'])
    # Adjust p-values
    df_pvalues['m_test_result'], df_pvalues['m_test_value'] = multipletests(df_pvalues.p_value, alpha=0.01, method='bonferroni')[:2]
    return df_pvalues[df_pvalues.m_test_result]
