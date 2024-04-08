from scipy.stats import hypergeom 
import pandas as pd
import numpy as np

# class CliqueAnalysis:
#     def __init__(self, sub_df, sub_df_control, df, clique, token_lookup):
#         self.sub_df = sub_df
#         self.sub_df_control = sub_df_control
#         self.df = df
#         self.clique = clique
#         self.token_lookup = token_lookup
#         self.analysis_result = {}
#         self.analysis_result_control = {}
#         self.pvalues = {}
    
#     def analyze(self):
#         self._analyze_descriptive_metrics()
#         self._analyze_influence_metrics()
#         # Add calls to other analysis methods here...
#         return [self.token_lookup[i] for i in self.clique], self.analysis_result, self.analysis_result_control, self.pvalues
    
#     def _analyze_descriptive_metrics(self):
#         # Implement the analysis for descriptive metrics
#         pass
    
#     def _analyze_influence_metrics(self):
#         # Implement the analysis for influence metrics
#         pass

#     # Implement other analysis methods...



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

from itertools import combinations


def clique_member_wallets_upper(clique, dataFrame):
    """
    Retrieve unique wallet addresses that hold tokens in at least two communities 
    within the provided clique.

    The function identifies wallets that hold positions (of any size) in 
    at least two distinct token communities that are part of a given token 
    projection clique. 

    Parameters:
    - clique (list): List of Token Smart Contract Addresses that form the clique.
    - dataFrame (pandas.DataFrame): DataFrame containing wallet addresses and their 
      associated tokens.

    Returns:
    - list: List of unique wallet addresses that are part of the token projection clique.

    Notes:
    The function uses combinations to create pairs of tokens from the given clique. 
    For each token pair, it retrieves the set of unique wallet addresses for each token 
    and then determines the intersection of these sets to identify wallets that hold 
    both tokens. All such wallets are considered part of the token projection clique.

    Example:
    If the clique consists of tokens [A, B], the function will find all wallet 
    addresses that hold both token A and token B.
    """
    
    clique_members = []

    # Calculate possible combinations for the given clique 
    for c in combinations(clique, 2):
                
        # Retrieve unique addresses for each token in the pair
        set1_addresses = dataFrame[dataFrame.token_address == c[0]].address.unique()
        set2_addresses = dataFrame[dataFrame.token_address == c[1]].address.unique()

        # Identify overlapping addresses between the two sets
        overlap_addresses_pair = set(set1_addresses) & set(set2_addresses)

        # Accumulate the overlapping addresses
        clique_members += overlap_addresses_pair
        
    return list(set(clique_members))



def clique_member_wallets_lower(clique, dataFrame):
    
    """
    Retrieve unique wallet addresses that hold ALL tokens in the provided clique.

    The function identifies wallets that possess all tokens within a given token 
    projection clique. These wallets have positions (of any size) in every token 
    community represented by the clique.

    Parameters:
    - clique (list): List of Token Smart Contract Addresses that form the clique.
    - dataFrame (pandas.DataFrame): DataFrame containing wallet addresses and their 
      associated tokens.

    Returns:
    - list: List of unique wallet addresses that hold all tokens in the clique.

    Notes:
    The function starts with a set of all wallet addresses for the first token in the 
    clique. For each subsequent token in the clique, it updates the set to the 
    intersection of the current set and the set of addresses for the current token. 
    This ensures that, by the end, the set only contains addresses that have all tokens 
    in the clique.

    Example:
    If the clique consists of tokens [A, B, C], the function will find all wallet 
    addresses that hold tokens A, B, AND C.
    """

    # Initialize with all wallets that hold the first token in the clique
    full_member_addresses = set(dataFrame[dataFrame.token_address == clique[0]].address.unique())

    # Intersect with wallets of each subsequent token in the clique
    for token in clique[1:]:
        token_addresses = set(dataFrame[dataFrame.token_address == token].address.unique())
        full_member_addresses &= token_addresses

        # Early exit: if no wallets hold all tokens so far, no need to check further
        if not full_member_addresses:
            return []

    return list(set(full_member_addresses))


