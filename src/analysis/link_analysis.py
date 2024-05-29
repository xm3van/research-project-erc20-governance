from scipy.stats import hypergeom 
import pandas as pd
import numpy as np

from src.utilities.metrics_and_tests import *
from src.utilities.utils import * 

from itertools import combinations

def link_member_wallets(link, dataFrame):
    
    """
    Retrieve unique wallet addresses that hold ALL tokens in the provided link.

    The function identifies wallets that possess all tokens within a given token 
    projection link. These wallets have positions (of any size) in every token 
    community represented by the link.

    Parameters:
    - link (list): List of Token Smart Contract Addresses that form the link.
    - dataFrame (pandas.DataFrame): DataFrame containing wallet addresses and their 
      associated tokens.

    Returns:
    - list: List of unique wallet addresses that hold all tokens in the link.

    Notes:
    The function starts with a set of all wallet addresses for the first token in the 
    link. For each subsequent token in the link, it updates the set to the 
    intersection of the current set and the set of addresses for the current token. 
    This ensures that, by the end, the set only contains addresses that have all tokens 
    in the link.

    Example:
    If the link consists of tokens [A, B, C], the function will find all wallet 
    addresses that hold tokens A, B, AND C.
    """

    # Initialize with all wallets that hold the first token in the link
    full_member_addresses = set(dataFrame[dataFrame.token_address == link[0]].address.unique())

    # Intersect with wallets of each subsequent token in the link
    for token in link[1:]:
        token_addresses = set(dataFrame[dataFrame.token_address == token].address.unique())
        full_member_addresses &= token_addresses

        # Early exit: if no wallets hold all tokens so far, no need to check further
        if not full_member_addresses:
            return []

    return list(set(full_member_addresses))


def analyse_link(sub_dataFrame, sub_dataFrame_control, dataFrame, link, token_lookup):
    """
    Analyze the characteristics of a link of wallets within a given DataFrame.

    Parameters:
    - sub_dataFrame (DataFrame): The DataFrame containing wallet data of given snapshot and a given link.
    - sub_dataFrame_control (DataFrame): The DataFrame acting as control for subDataFrame containing wallet data of given snapshot and a members of given link.
    - dataFrame (DataFrame): The DataFrame containing all wallet data of given snapshot.
    - link (list): The list of tokens or addresses representing the link.
    - token_lookup (dict): A dictionary to map tokens or addresses to human-readable labels.

    Returns:
    - link_name (list): Human-readable names of the tokens or addresses in the link.
    - result (dict): A dictionary containing various metrics and characteristics of the link.
    - pvalues (dict): A dictionary containing pvalues compared to control.

    """
    
    analysis_result = {} 
    analysis_result_control = {} 
    pvalues = {}
    
    # ====================================================================================== # 
    
    # Metric 1: Descriptive
    
    # MeEtric 1.1: Number of Wallets
    ## Analysis for 1.1
    size_link = sub_dataFrame.address.nunique()
    size_linkC = sub_dataFrame_control.address.nunique()

    ## Logging for 1.1
    analysis_result['size_link'] = size_link
    analysis_result_control['size_link'] = size_linkC
    

    # Metric 1.2: Basic Description Presence
    ## Analysis for 1.2
    no_token_communities_snapshot = dataFrame.token_address.nunique()
    no_token_communities_in_link = len(link)
    no_token_communities_not_in_link = no_token_communities_snapshot - no_token_communities_in_link
    
    no_token_communities_in_linkC = len(link)
    no_token_communities_not_in_linkC = no_token_communities_snapshot - no_token_communities_in_linkC
    

    # ## logging for 1.2
    # analysis_result['no_communities_link'] = no_token_communities_in_link
    # analysis_result['pct_presence'] = no_token_communities_in_link/ no_token_communities_snapshot #meaningless

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
    normalised_pct_supply_internal = sub_dataFrame[sub_dataFrame.token_address.isin(link) == True].pct_supply.apply(lambda x: x / no_token_communities_in_link)
    internal_influence = normalised_pct_supply_internal.sum()
    
    ## Validation for 2.1
    normalised_pct_supply_internalC = sub_dataFrame_control[sub_dataFrame_control.token_address.isin(link) == True].pct_supply.apply(lambda x: x / no_token_communities_in_linkC)
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
    normalised_pct_supply_external = sub_dataFrame[sub_dataFrame.token_address.isin(link) != True].pct_supply.apply(lambda x: x / no_token_communities_not_in_link)
    external_influence = normalised_pct_supply_external.sum()
    
     ## Validation for 2.2
    normalised_pct_supply_externalC = sub_dataFrame_control[sub_dataFrame_control.token_address.isin(link) != True].pct_supply.apply(lambda x: x / no_token_communities_in_linkC)
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

    # Metric 3.0: Wealth link
    ## Analysis for Metric 3.0
    wealth_link = sub_dataFrame['value_usd'].sum()
    
    ## Validation
    total_wealth_pval = t_test(sub_dataFrame['value_usd'], sub_dataFrame_control['value_usd'], alternative='greater')
    
    ## Logging for Metric 3.0
    analysis_result['wealth_link'] = wealth_link
    analysis_result_control['wealth_link'] = sub_dataFrame_control['value_usd'].sum()
    pvalues['wealth_link'] = total_wealth_pval
    
    

    # Metric 3.0.0: Gini Coefficient 
    # Calculate and log Gini coefficient for wealth_link
    gini_wealth_link = gini(sub_dataFrame['value_usd'])
    gini_wealth_linkC = gini(sub_dataFrame_control['value_usd'])

    
    ## Validation for 3.0.0
    ### H0: The null hypothesis is that there is no significant difference between the Gini coefficients of the two groups.
    ### H1: Group 1 has a more equal distribution (lower Gini coefficient) than Group 2. 
    gini_wealth_link_pval = one_tailed_gini_t_test(sub_dataFrame['value_usd'], sub_dataFrame_control['value_usd'], alpha=0.05, direction="lower")

    ## Logging for Metric 3.0.0
    analysis_result['gini_wealth_link'] = gini_wealth_link
    analysis_result_control['gini_wealth_link'] = gini_wealth_linkC
    pvalues['gini_wealth_link'] = gini_wealth_link_pval
    
    
    
    # Metric 3.0.1: Median Wealth Level in link
    ## Analysis for Metric 3.0.1
    median_wealth_level_link = sub_dataFrame.groupby('address')['value_usd'].sum().median()
    median_wealth_level_linkC = sub_dataFrame_control.groupby('address')['value_usd'].sum().median()

    
    ## Validation for Metric 3.0.1
    median_wealth_level_link_pval = median_t_test(sub_dataFrame.groupby('address')['value_usd'].sum(), 
                                                    sub_dataFrame_control.groupby('address')['value_usd'].sum(),
                                                    alternative='greater')
    
    ## Logging for Metric 3.0.1
    analysis_result['median_wealth_level_link'] = median_wealth_level_link
    analysis_result_control['median_wealth_level_link'] = median_wealth_level_linkC
    pvalues['median_wealth_level_link'] = median_wealth_level_link_pval
    
    

    # Metric 3.0.2: Median Number of Assets Held in link
    ## Analysis for Metric 3.0.2
    median_no_assets_link = sub_dataFrame.groupby('address')['token_address'].count().median()
    median_no_assets_linkC = sub_dataFrame_control.groupby('address')['token_address'].count().median()

    
    ## Validation for Metric 3.0.2
    median_no_assets_link_pval = median_t_test(sub_dataFrame.groupby('address')['token_address'].count(), 
                                                 sub_dataFrame_control.groupby('address')['token_address'].count(), 
                                                 alternative='greater')
    
    ## Logging for Metric 3.0.2
    analysis_result['median_no_assets_link'] = median_no_assets_link
    analysis_result_control['median_no_assets_link'] = median_no_assets_linkC
    pvalues['median_no_assets_link'] = median_no_assets_link_pval
    
    
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
    
    
    # Metric 4.0: Max Influence Label
    analysis_result['max_influence_label_distribution'] = dict(sub_dataFrame.groupby(['label'])['pct_supply'].sum()/ no_token_communities_snapshot)
    analysis_result_control['max_influence_label_distribution'] = dict(sub_dataFrame_control.groupby(['label'])['pct_supply'].sum()/ no_token_communities_snapshot)

    
    ###
    # TO-DO: 
    # Add in validation 
    # Add in max label internal influence 
    # Add in max label external influence
    
    # ====================================================================================== # 


    # Add more analysis metrics as needed...
    

    # ====================================================================================== # 


    # Human readable link
    link_name = [token_lookup[i] for i in link]

    return link_name, analysis_result, analysis_result_control, pvalues
