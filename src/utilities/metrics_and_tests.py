import numpy as np
import pandas as pd

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




import scipy.stats as stats
import numpy as np

def one_tailed_gini_t_test(group1_data, group2_data, alpha=0.05, direction="lower"):
    """
    Perform a one-tailed t-test to compare Gini coefficients between two groups, testing the hypothesis that Group 1 has a more equal distribution than Group 2.

    Parameters:
    - group1_data: List or array containing data for the first group.
    - group2_data: List or array containing data for the second group.
    - alpha: Significance level (default is 0.05).
    - direction: Direction of the test ("lower" for lower Gini coefficient in Group 1, "higher" for higher Gini coefficient in Group 1).

    Returns:
    - t_statistic: The t-statistic for the t-test.
    - p_value: The one-tailed p-value for the t-test.
    - significant: A boolean indicating whether the difference is statistically significant.
    """
    
    if len(group1_data) < 15 or len(group2_data) < 15: 
        # print('Insufficient Sample Size to ensure statistical validitiy of t-test') 
        return np.nan
    
    # Calculate Gini coefficients for each group
    gini_group1 = gini(group1_data)
    gini_group2 = gini(group2_data)

    # Sample sizes for each group
    n_group1 = len(group1_data)
    n_group2 = len(group2_data)

    # Calculate the standard errors for each group
    se_group1 = (1 / (2 * n_group1)) ** 0.5
    se_group2 = (1 / (2 * n_group2)) ** 0.5

    # Calculate the t-statistic
    t_statistic = (gini_group1 - gini_group2) / ((se_group1 ** 2 / n_group1) + (se_group2 ** 2 / n_group2)) ** 0.5

    # Calculate the degrees of freedom
    degrees_of_freedom = n_group1 + n_group2 - 2  # Explanation: (n1 + n2 - 2) degrees of freedom for the t-test

    # Calculate the one-tailed p-value based on the specified direction
    if direction == "lower":
        p_value = stats.t.cdf(t_statistic, df=degrees_of_freedom)
    elif direction == "higher":
        p_value = 1 - stats.t.cdf(t_statistic, df=degrees_of_freedom)
    else:
        raise ValueError("Direction must be 'lower' or 'higher'.")

    # Determine if the result is statistically significant
    significant = p_value < alpha

    return p_value




def t_test(sample_data, control_data, alternative='greater'): 
    
    # Test applicability of t-test
    
    ### Independence: The data in both groups should be independent of each other. - True 
    
    if len(sample_data) < 15 or len(control_data) < 15: 
        print('Insufficient Sample Size to ensure statistical validitiy of t-test') 
        return 1
    
    elif len(sample_data) < 50 or len(control_data) < 50: 
    # Only test if sufficiently large sample size (typically greater than 30-40 observations per group) 
    # the central limit theorem often comes into play.
    
        ### Normality: The data in each group should follow a roughly normal distribution.
        _, p_value = stats.shapiro(sample_data)
        _, p_value_control = stats.shapiro(control_data)

        if p_value < 0.05 or p_value_control < 0.05: 

            print(f"Significantly deviates from a normal distribution: Sample: {p_value} || Control: {p_value_control}") 
            return 1

        ### Homogeneity of Variances: The variances in both groups should be approximately equal. You can check this using statistical tests like Levene's test or by visual inspection of boxplots.
        _, p_value = stats.levene(sample_data, control_data)

        if p_value < 0.05: 

            print(f"The variances are not approximately equal: Sample: {p_value} || Control: {p_value_control}") 
            return 1
        
    
    # Apply t-test 
    ### For all metrics we expect 
    
    # Perform the one-sided t-test
    _, p_value = stats.ttest_ind(sample_data, control_data, alternative=alternative)
    return p_value



import scipy.stats as stats

def median_t_test(group1_data, group2_data, alternative='greater', alpha=0.05):
    """
    Perform a Mann-Whitney U test (t-test for the median) to compare medians between two groups.

    Parameters:
    - group1_data: List or array containing data for the first group.
    - group2_data: List or array containing data for the second group.
    - alpha: Significance level (default is 0.05).

    Returns:
    - u_statistic: The Mann-Whitney U statistic.
    - p_value: The two-tailed p-value for the test.
    - significant: A boolean indicating whether the difference is statistically significant.
    """
    if len(sample_data) < 15: 
        # print('Insufficient Sample Size to ensure statistical validitiy of t-test') 
        return 1
    
    # Perform Mann-Whitney U test
    _, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative=alternative)

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

def jaccard_similarity(matrix1, matrix2):
    set1 = set(np.reshape(matrix1, -1))
    set2 = set(np.reshape(matrix2, -1))
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity