import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def jaccard_similarity(g1, g2):
    """
    Calculate Jaccard Similarity between two graphs.

    Parameters:
    g1 (networkx.Graph): The first graph.
    g2 (networkx.Graph): The second graph.

    Returns:
    float: Jaccard Similarity between the two graphs.
    """
    intersection = len(set(g1.edges()).intersection(set(g2.edges())))
    union = len(set(g1.edges()).union(set(g2.edges())))
    return intersection / union if union != 0 else 0


def gini(array):
    """
    Calculate the Gini coefficient of an array representing income or wealth distribution.

    Args:
        array (numpy.ndarray or list): 1D array-like object containing non-negative values.

    Returns:
        float: Gini coefficient value between 0 and 1.
    """
    # Convert input to a NumPy array
    array = np.asarray(array, dtype=float)

    # Check if array is empty
    if len(array) == 0:
        return np.nan

    # Ensure all values are non-negative
    if np.any(array < 0):
        raise ValueError("Input array contains negative values.")

    # Handle the case of a single element
    if len(array) == 1:
        return 0.0

    # Sort the array in ascending order
    array = np.sort(array)

    # Calculate the Gini coefficient
    n = len(array)
    index = np.arange(1, n + 1)
    gini_coefficient = (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

    return gini_coefficient



def permutation_test(full_sample, cdw_sample, num_permutations=1000):
    n_cdw = len(cdw_sample)
    full_mean = np.mean(full_sample)
    cdw_mean = np.mean(cdw_sample)
    
    # Calculate observed test statistic M_CDW
    M_CDW = cdw_mean - full_mean
    
    # Generate permutation samples and calculate M_k
    M_k_values = []
    for _ in range(num_permutations):
        perm_sample = np.random.choice(full_sample, size=n_cdw, replace=False)
        perm_mean = np.mean(perm_sample)
        M_k = perm_mean - full_mean
        M_k_values.append(M_k)
    
    # Convert M_k_values to a numpy array for easier quantile calculation
    M_k_values = np.array(M_k_values)
    
    # Calculate the 95th percentile of the permutation test statistics
    threshold = np.percentile(M_k_values, 95)
    
    # Calculate p-value as the proportion of M_k values greater than or equal to M_CDW
    p_value = np.mean(M_k_values >= M_CDW)
    
    return M_CDW, threshold, p_value

# # Example usage
# full_sample = np.random.normal(loc=10, scale=2, size=1000)  # All token holders
# cdw_sample = np.random.normal(loc=12, scale=2, size=30)    # CDW token holders

# # Perform the permutation test
# M_CDW, threshold, p_value = permutation_test(full_sample, cdw_sample)
# print(f"Observed M_CDW: {M_CDW}")
# print(f"95th percentile threshold: {threshold}")
# print(f"P-value: {p_value}")



import numpy as np
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower

def t_test(sample_data, control_data, alternative='greater'):
    # Check independence assumption (this needs to be ensured during data collection)
    # Assuming independence for the purpose of this function
    
    # Perform power analysis to determine if sample size is sufficient
    effect_size = np.abs(np.mean(sample_data) - np.mean(control_data)) / np.sqrt((np.var(sample_data) + np.var(control_data)) / 2)
    power_analysis = TTestIndPower()
    required_n = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, ratio=len(control_data)/len(sample_data))
    
    if len(sample_data) < required_n or len(control_data) < required_n:
        print('Insufficient Sample Size to ensure statistical validity of t-test')
        return 1, None
    
    # For small sample sizes (< 30), check for normality
    if len(sample_data) < 30 or len(control_data) < 30:
        # Shapiro-Wilk test for normality
        _, p_value_sample = stats.shapiro(sample_data)
        _, p_value_control = stats.shapiro(control_data)

        if p_value_sample < 0.05 or p_value_control < 0.05:
            print(f"Data significantly deviates from normal distribution: Sample: {p_value_sample} || Control: {p_value_control}")
            # Perform Mann-Whitney U test as a non-parametric alternative
            u_statistic, p_value = stats.mannwhitneyu(sample_data, control_data, alternative=alternative)
            return u_statistic, p_value

    # Check for homogeneity of variances using Levene's test
    _, p_value_levene = stats.levene(sample_data, control_data)

    if p_value_levene < 0.05:
        print(f"The variances are not approximately equal: Levene's Test p-value: {p_value_levene}")
        # Perform Welch's t-test
        t_statistic, p_value = stats.ttest_ind(sample_data, control_data, equal_var=False, alternative=alternative)
    else:
        # Perform Student's t-test
        t_statistic, p_value = stats.ttest_ind(sample_data, control_data, equal_var=True, alternative=alternative)

    return p_value


# def t_test(sample_data, control_data, alternative='greater'): 
    
#     # Test applicability of t-test
    
#     ### Independence: The data in both groups should be independent of each other. - True 
    
#     if len(sample_data) < 15 or len(control_data) < 15: 
#         print('Insufficient Sample Size to ensure statistical validitiy of t-test') 
#         return 1
    
#     elif len(sample_data) < 30 or len(control_data) < 30: 
#     # Only test if sufficiently large sample size (typically greater than 30-40 observations per group) 
#     # the central limit theorem often comes into play.
    
#         ### Normality: The data in each group should follow a roughly normal distribution.
#         _, p_value = stats.shapiro(sample_data)
#         _, p_value_control = stats.shapiro(control_data)

#         if p_value < 0.05 or p_value_control < 0.05: 

#             print(f"Significantly deviates from a normal distribution: Sample: {p_value} || Control: {p_value_control}") 
#             return 1

#         ### Homogeneity of Variances: The variances in both groups should be approximately equal. You can check this using statistical tests like Levene's test or by visual inspection of boxplots.
#         _, p_value = stats.levene(sample_data, control_data)

#         if p_value < 0.05: 

#             print(f"The variances are not approximately equal: Sample: {p_value} || Control: {p_value_control}") 
#             # Perform the  Welch’s t-test
#             _, p_value = stats.ttest_ind(sample_data, control_data, equal_var=False,  alternative=alternative)
#         else: 
#              _, p_value = stats.ttest_ind(sample_data, control_data, equal_var=True,  alternative=alternative)


#     return p_value




# import scipy.stats as stats

# def median_t_test(group1_data, group2_data, alternative='greater'):
#     """
#     Perform a Mann-Whitney U test (t-test for the median) to compare medians between two groups.

#     Parameters:
#     - group1_data: List or array containing data for the first group.
#     - group2_data: List or array containing data for the second group.
#     - alpha: Significance level (default is 0.05).

#     Returns:
#     - u_statistic: The Mann-Whitney U statistic.
#     - p_value: The two-tailed p-value for the test.
#     - significant: A boolean indicating whether the difference is statistically significant.
#     """
#     # Check for sufficient variability within each group
#     if len(np.unique(group1_data)) < 2 or len(np.unique(group2_data)) < 2:
#         print('Insufficient variability within one or both groups for Mann-Whitney U test.')
#         return 1
    
#     # Perform Mann-Whitney U test
#     _, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative=alternative)

#     return p_value



def pval_to_significance(pval):
    """
    Convert a p-value into a significance symbol based on certain conditions.

    Parameters:
    - pval: The p-value to be converted.

    Returns:
    - significance_symbol: The significance symbol ('***', '**', '*', or '') based on the p-value.
    """
    if pval < 0.01:
        significance_symbol = '***'
    elif pval < 0.05:
        significance_symbol = '**'
    elif pval < 0.1:
        significance_symbol = '*'
    else:
        significance_symbol = ''
    
    return significance_symbol

