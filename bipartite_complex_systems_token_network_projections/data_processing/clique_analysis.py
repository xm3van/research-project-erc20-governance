from scipy.stats import hypergeom 
import pandas as pd
import numpy as np

def validate_token_link(address1, address2, data_set, pop_size):

    # get unique addresses 
    token1_uniqa = data_set[data_set.token_address == address1].address.unique()
    token2_uniqa = data_set[data_set.token_address == address2].address.unique()

    # calcluate intersection 
    token1_token2_uniqa_intersection = np.intersect1d(token1_uniqa,token2_uniqa, assume_unique=True)

    # calcualte number 
    len_token1 = len(token1_uniqa)
    len_token2 = len(token2_uniqa)
    len_intersection = len(token1_token2_uniqa_intersection)

    # calculate hyptoge

    # Define the parameters of the distribution
    M = pop_size  # population size
    n = len_token1  # number of draws
    K = len_token2  # number of successes in population
    x = len_intersection    # number of successes in draws

    # Compute the cumulative probability of obtaining at most x successes
    pvalue = 1 - hypergeom.cdf(x, M, n, K)
    
    # print(f'token_address {address1} has {len_token1} Unique Addresses | token_address {address2} has {len_token2} Unique Addresses | Intersection: {len_intersection}, p value: {pvalue}')

    return pvalue

# # Demo
# # test tokens 
# test_token1 = '0x111111111117dc0aa78b770fa6a738034120c302'
# test_token2 = '0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e'

# # calc pop_size
# pop_size = len(ddf.address.unique()) 

# # main
# main(test_token1,test_token2, data_set, pop_size)