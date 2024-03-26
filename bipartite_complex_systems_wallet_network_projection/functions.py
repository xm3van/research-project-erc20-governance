import pandas as pd 
from scipy.stats import hypergeom 
import numpy as np

def hypergeom_wallets(wallet1, wallet2, pop_size, df):
    """
    Computes the p-value of the intersection between the unique tokens held by two wallets using a hypergeometric
    distribution. 

    Parameters:
        - wallet1 (str): The address of the first wallet.
        - wallet2 (str): The address of the second wallet.
        - pop_size (int): The total number of possible tokens that can be held in the sample.
        - df (pandas.DataFrame): The data frame containing the token holdings for all wallets.

    Returns:
        The p-value of the intersection between the unique tokens held by the two wallets.
    """
    ...

    # unique token a given address holds
    wallet1_uniqT = df[df.address == wallet1].token_address.unique()
    wallet2_uniqT = df[df.address == wallet2].token_address.unique()

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
    
    return wallet1, wallet2, pvalue
