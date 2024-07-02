import numpy as np


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


def permutation_test(cdw_sample, full_sample, method='mean', alternative='greater', num_permutations=1000, min_sample_size=30):
    if len(full_sample) < min_sample_size or len(cdw_sample) < min_sample_size:
        return 1.0  # non-significant due to small sample size

    n_cdw = len(cdw_sample)
    
    # Calculate observed statistic
    if method == 'mean':
        observed_stat_cdw = np.mean(cdw_sample)
        observed_stat_full = np.mean(full_sample)
    elif method == 'median':
        observed_stat_cdw = np.median(cdw_sample)
        observed_stat_full = np.median(full_sample)
    elif method == 'gini':
        observed_stat_cdw = gini(cdw_sample)
        observed_stat_full = gini(full_sample)
    else:
        raise ValueError("Method must be 'mean', 'median', or 'gini'")

    observed_diff = observed_stat_cdw - observed_stat_full
    
    # Generate permutation samples and calculate statistic differences
    combined_sample = np.concatenate([cdw_sample, full_sample])
    perm_diffs = []
    for _ in range(num_permutations):
        np.random.shuffle(combined_sample)
        perm_cdw = combined_sample[:n_cdw]
        perm_full = combined_sample[n_cdw:]
        
        if method == 'mean':
            perm_stat_cdw = np.mean(perm_cdw)
            perm_stat_full = np.mean(perm_full)
        elif method == 'median':
            perm_stat_cdw = np.median(perm_cdw)
            perm_stat_full = np.median(perm_full)
        elif method == 'gini':
            perm_stat_cdw = gini(perm_cdw)
            perm_stat_full = gini(perm_full)
        
        perm_diff = perm_stat_cdw - perm_stat_full
        perm_diffs.append(perm_diff)
    
    perm_diffs = np.array(perm_diffs)
    
    if alternative == 'greater':
        # Calculate p-value for greater alternative
        p_value = np.mean(perm_diffs >= observed_diff)
    elif alternative == 'lower':
        # Calculate p-value for less alternative
        p_value = np.mean(perm_diffs <= observed_diff)
    else:
        raise ValueError("Alternative hypothesis must be 'greater' or 'less'")
    
    return p_value





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

