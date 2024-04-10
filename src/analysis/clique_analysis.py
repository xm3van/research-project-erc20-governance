from scipy.stats import hypergeom 
import pandas as pd
import numpy as np

from src.utilities.metrics_and_tests import *
from src.utilities.utils import * 

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


def analyze_clique(sub_dataFrame, sub_dataFrame_control, dataFrame, clique, token_lookup):
    """
    Analyze the characteristics of a clique of wallets within a given DataFrame.

    Parameters:
    - sub_dataFrame (DataFrame): The DataFrame containing wallet data of given snapshot and a given clique.
    - sub_dataFrame_control (DataFrame): The DataFrame acting as control for subDataFrame containing wallet data of given snapshot and a members of given clique.
    - dataFrame (DataFrame): The DataFrame containing all wallet data of given snapshot.
    - clique (list): The list of tokens or addresses representing the clique.
    - token_lookup (dict): A dictionary to map tokens or addresses to human-readable labels.

    Returns:
    - clique_name (list): Human-readable names of the tokens or addresses in the clique.
    - result (dict): A dictionary containing various metrics and characteristics of the clique.
    - pvalues (dict): A dictionary containing pvalues compared to control.

    """
    
    analysis_result = {} 
    analysis_result_control = {} 
    pvalues = {}
    
    # ====================================================================================== # 
    
    # Metric 1: Descriptive
    
    # MeEtric 1.1: Number of Wallets
    ## Analysis for 1.1
    size_clique = sub_dataFrame.address.nunique()
    size_cliqueC = sub_dataFrame_control.address.nunique()

    ## Logging for 1.1
    analysis_result['size_clique'] = size_clique
    analysis_result_control['size_clique'] = size_cliqueC
    

    # Metric 1.2: Basic Description Presence
    ## Analysis for 1.2
    no_token_communities_snapshot = dataFrame.token_address.nunique()
    no_token_communities_in_clique = len(clique)
    no_token_communities_not_in_clique = no_token_communities_snapshot - no_token_communities_in_clique
    
    no_token_communities_in_cliqueC = len(clique)
    no_token_communities_not_in_cliqueC = no_token_communities_snapshot - no_token_communities_in_cliqueC
    

    # ## logging for 1.2
    # analysis_result['no_communities_clique'] = no_token_communities_in_clique
    # analysis_result['pct_presence'] = no_token_communities_in_clique/ no_token_communities_snapshot #meaningless

    # ====================================================================================== # 
    
    # Metric 2: Metrics around Pct_supply
    
    # Metric 2.0: Total Influence
    ## Analysis for 2.0 
    normalised_pct_supply = sub_dataFrame.pct_supply.apply(lambda x: x / no_token_communities_snapshot)
    total_influence = normalised_pct_supply.sum()
    
    ## Validation for 2.0
    normalised_pct_supplyC = sub_dataFrame_control.pct_supply.apply(lambda x: x / no_token_communities_snapshot)
    total_influence_pval = t_test(normalised_pct_supply, normalised_pct_supplyC, alternative='greater')
    
    ## Logging for 2.0
    analysis_result['total_influence'] = total_influence
    analysis_result_control['total_influence'] = normalised_pct_supplyC.sum()
    pvalues['total_influence'] = total_influence_pval

    
    # Metric 2.0.0: Normalised Total Influence Gini Coefficient
    ## Analysis for 2.0.0
    gini_total_influence = gini(normalised_pct_supply)
    gini_total_influenceC = gini(normalised_pct_supplyC)


    ## Validation for 2.0.0
    ### H0: The null hypothesis is that there is no significant difference between the Gini coefficients of the two groups.
    ### H1: Group 1 has a more equal distribution (lower Gini coefficient) than Group 2. 
    gini_total_influence_pval = one_tailed_gini_t_test(normalised_pct_supply, normalised_pct_supplyC, alpha=0.05, direction="lower")
    
    ## Logging for 2.0.0
    analysis_result['gini_total_influence'] = gini_total_influence
    analysis_result_control['gini_total_influence'] = gini_total_influenceC
    pvalues['gini_total_influence'] = gini_total_influence_pval
    

    
    # Metric 2.1: Internal influence
    ## Analysis for 2.1
    normalised_pct_supply_internal = sub_dataFrame[sub_dataFrame.token_address.isin(clique) == True].pct_supply.apply(lambda x: x / no_token_communities_in_clique)
    internal_influence = normalised_pct_supply_internal.sum()
    
    ## Validation for 2.1
    normalised_pct_supply_internalC = sub_dataFrame_control[sub_dataFrame_control.token_address.isin(clique) == True].pct_supply.apply(lambda x: x / no_token_communities_in_cliqueC)
    total_internal_influence_pval = t_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, alternative='greater')
    
    ## Logging for 2.1
    analysis_result['internal_influence'] = internal_influence
    analysis_result_control['internal_influence'] = normalised_pct_supply_internalC.sum()
    pvalues['internal_influence'] = total_internal_influence_pval
    
    

    # Metric 2.1.0: Normalised Internal Influence Gini Coefficient
    ## Analysis for 2.1.0
    gini_internal_influence = gini(normalised_pct_supply_internal)
    gini_internal_influenceC = gini(normalised_pct_supply_internalC)

    
    ## Validation for 2.1.0
    ### H0: The null hypothesis is that there is no significant difference between the Gini coefficients of the two groups.
    ### H1: Group 1 has a more equal distribution (lower Gini coefficient) than Group 2. 
    gini_internal_influence_pval = one_tailed_gini_t_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, alpha=0.05, direction="lower")

    ## Logging for 2.1.0
    analysis_result['gini_internal_influence'] = gini_internal_influence
    analysis_result_control['gini_internal_influence'] = gini_internal_influenceC
    pvalues['gini_internal_influence'] = gini_internal_influence_pval
   
    
    
    # Metric 2.2: External influence
    ## Analysis for 2.2
    normalised_pct_supply_external = sub_dataFrame[sub_dataFrame.token_address.isin(clique) != True].pct_supply.apply(lambda x: x / no_token_communities_not_in_clique)
    external_influence = normalised_pct_supply_external.sum()
    
     ## Validation for 2.2
    normalised_pct_supply_externalC = sub_dataFrame_control[sub_dataFrame_control.token_address.isin(clique) != True].pct_supply.apply(lambda x: x / no_token_communities_in_cliqueC)
    total_external_influence_pval = t_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, alternative='greater')
    
     ## Logging for 2.1
    analysis_result['external_influence'] = external_influence
    analysis_result_control['external_influence'] =normalised_pct_supply_externalC.sum()
    pvalues['external_influence'] = total_external_influence_pval
    
    

    # Metric 2.2.0: Normalised External Gini Coefficient
    ## Analysis for 2.2.0
    gini_external_influence = gini(normalised_pct_supply_external)
    gini_external_influenceC = gini(normalised_pct_supply_externalC)

    ## Validation for 2.2.0
    ### H0: The null hypothesis is that there is no significant difference between the Gini coe fficients of the two groups.
    ### H1: Group 1 has a more equal distribution (lower Gini coefficient) than Group 2. 
    gini_external_influence_pval = one_tailed_gini_t_test(normalised_pct_supply_external, normalised_pct_supply_externalC, alpha=0.05, direction="lower")

    ## Logging for  2.2.0
    analysis_result['gini_external_influence'] = gini_external_influence
    analysis_result_control['gini_external_influence'] = gini_external_influenceC
    pvalues['gini_external_influence'] = gini_external_influence_pval
    
    
    # ====================================================================================== # 

    # Metric 3: Portfolio Value Distribution
    ## Analysis for Metric 3

    # Metric 3.0: Wealth Clique
    ## Analysis for Metric 3.0
    wealth_clique = sub_dataFrame['value_usd'].sum()
    
    ## Validation
    total_wealth_pval = t_test(sub_dataFrame['value_usd'], sub_dataFrame_control['value_usd'], alternative='greater')
    
    ## Logging for Metric 3.0
    analysis_result['wealth_clique'] = wealth_clique
    analysis_result_control['wealth_clique'] = sub_dataFrame_control['value_usd'].sum()
    pvalues['wealth_clique'] = total_wealth_pval
    
    

    # Metric 3.0.0: Gini Coefficient 
    # Calculate and log Gini coefficient for wealth_clique
    gini_wealth_clique = gini(sub_dataFrame['value_usd'])
    gini_wealth_cliqueC = gini(sub_dataFrame_control['value_usd'])

    
    ## Validation for 3.0.0
    ### H0: The null hypothesis is that there is no significant difference between the Gini coefficients of the two groups.
    ### H1: Group 1 has a more equal distribution (lower Gini coefficient) than Group 2. 
    gini_wealth_clique_pval = one_tailed_gini_t_test(sub_dataFrame['value_usd'], sub_dataFrame_control['value_usd'], alpha=0.05, direction="lower")

    ## Logging for Metric 3.0.0
    analysis_result['gini_wealth_clique'] = gini_wealth_clique
    analysis_result_control['gini_wealth_clique'] = gini_wealth_cliqueC
    pvalues['gini_wealth_clique'] = gini_wealth_clique_pval
    
    
    
    # Metric 3.0.1: Median Wealth Level in Clique
    ## Analysis for Metric 3.0.1
    median_wealth_level_clique = sub_dataFrame.groupby('address')['value_usd'].sum().median()
    median_wealth_level_cliqueC = sub_dataFrame_control.groupby('address')['value_usd'].sum().median()

    
    ## Validation for Metric 3.0.1
    median_wealth_level_clique_pval = median_t_test(sub_dataFrame.groupby('address')['value_usd'].sum(), 
                                                    sub_dataFrame_control.groupby('address')['value_usd'].sum(),
                                                    alternative='greater')
    
    ## Logging for Metric 3.0.1
    analysis_result['median_wealth_level_clique'] = median_wealth_level_clique
    analysis_result_control['median_wealth_level_clique'] = median_wealth_level_cliqueC
    pvalues['median_wealth_level_clique'] = median_wealth_level_clique_pval
    
    

    # Metric 3.0.2: Median Number of Assets Held in Clique
    ## Analysis for Metric 3.0.2
    median_no_assets_clique = sub_dataFrame.groupby('address')['token_address'].count().median()
    median_no_assets_cliqueC = sub_dataFrame_control.groupby('address')['token_address'].count().median()

    
    ## Validation for Metric 3.0.2
    median_no_assets_clique_pval = median_t_test(sub_dataFrame.groupby('address')['token_address'].count(), 
                                                 sub_dataFrame_control.groupby('address')['token_address'].count(), 
                                                 alternative='greater')
    
    ## Logging for Metric 3.0.2
    analysis_result['median_no_assets_clique'] = median_no_assets_clique
    analysis_result_control['median_no_assets_clique'] = median_no_assets_cliqueC
    pvalues['median_no_assets_clique'] = median_no_assets_clique_pval
    
    
    # ====================================================================================== # 


    # Metric 4: Labels
    ## Analysis for Metric 4
    sub_dataFrame.label.fillna('other_contracts', inplace=True)
    sub_dataFrame_control.label.fillna('other_contracts', inplace=True)
    
    # Metric 4.0: Max Influence Label
    max_influence_label = sub_dataFrame.groupby(['label'])['pct_supply'].sum().idxmax()
    max_influence_labelC = sub_dataFrame_control.groupby(['label'])['pct_supply'].sum().idxmax()

    ## Analysis for Metric 4.0
    max_influence_label_value = sub_dataFrame.groupby(['label'])['pct_supply'].sum().max()/ no_token_communities_snapshot
    max_influence_label_valueC = sub_dataFrame_control.groupby(['label'])['pct_supply'].sum().max()/ no_token_communities_snapshot

    
    ## Validation for Metric 4.0 
    sample = sub_dataFrame[sub_dataFrame.label==max_influence_label].pct_supply
    control = sub_dataFrame_control[sub_dataFrame_control.label==max_influence_label].pct_supply
    
    ## Logging for Metric 4.0
    analysis_result['max_influence_label'] = (str(max_influence_label), max_influence_label_value)
    analysis_result_control['max_influence_label'] = (str(max_influence_labelC), max_influence_label_valueC)
    pvalues['max_influence_label'] = t_test(sample, control, alternative='greater')

    
    ###
    # TO-DO: 
    # Add in validation 
    # Add in max label internal influence 
    # Add in max label external influence
    
    # ====================================================================================== # 


    # Add more analysis metrics as needed...
    

    # ====================================================================================== # 


    # Human readable Clique
    clique_name = [token_lookup[i] for i in clique]

    return clique_name, analysis_result, analysis_result_control, pvalues
