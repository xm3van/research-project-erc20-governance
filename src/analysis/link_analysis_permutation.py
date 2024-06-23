import pandas as pd
import numpy as np
import pickle
from os.path import join
from itertools import combinations
from src.utilities.metrics_and_tests import *
from src.utilities.utils import *

class LinkData:
    """
    Class to load and manage link data.

    Attributes:
        data_path (str): The file path to the link data.
        links (dict): The loaded link data.
        metric_names (list): List of metric names.

    Methods:
        load_links(): Loads link data from a pickle file.
        get_metric_names(): Retrieves the metric names from the link data.
        get_metric_data(group, metric_name): Retrieves the metric data for a specific group and metric.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.links = self.load_links()
        self.metric_names = self.get_metric_names()

    def load_links(self):
        """
        Loads link data from the specified file path.

        Returns:
            dict: The loaded link data.
        """
        with open(self.data_path, 'rb') as handle:
            links = pickle.load(handle)
        return links

    def get_metric_names(self):
        """
        Retrieves the metric names from the link data.

        Returns:
            list: A list of metric names.
        """
        example_key = next(iter(self.links['sample']))
        metrics_example = self.links['sample'][example_key]
        first_link_key = next(iter(metrics_example.keys()))
        metric_names = list(metrics_example[first_link_key].keys())
        return metric_names

    def get_metric_data(self, group, metric_name):
        """
        Retrieves the metric data for a specific group and metric.

        Args:
            group (str): The group name.
            metric_name (str): The metric name.

        Returns:
            pd.DataFrame: A DataFrame containing the metric data.
        """
        data = []

        for date, snapshot_data in self.links[group].items():
            for link_name, metrics in snapshot_data.items():
                try:
                    metric_value = metrics[metric_name]
                except KeyError:
                    metric_value = np.nan
                data.append({'Date': date, 'Link Name': link_name, metric_name: metric_value})

        df_metric = pd.DataFrame(data)
        df_metric = df_metric.pivot(index='Link Name', columns='Date', values=metric_name)

        return df_metric

class LinkAnalysis:
    """
    Class to perform analysis on link data.

    Attributes:
        link (list): List of token addresses in the link.
        dataFrame (pd.DataFrame): The main DataFrame containing token data.
        sub_dataFrame (pd.DataFrame): Subset of the main DataFrame for the link.
        sub_dataFrame_control (pd.DataFrame): Control subset of the DataFrame.
        token_lookup (dict): Dictionary for token address to token name mapping.
        analysis_result (dict): Dictionary to store analysis results.
        analysis_result_control (dict): Dictionary to store control analysis results.
        pvalues (dict): Dictionary to store p-values of the statistical tests.

    Methods:
        link_member_wallets(): Retrieves unique wallet addresses that hold all tokens in the provided link.
        analyze_link(): Analyzes the characteristics of the link.
        _analyze_descriptive_metrics(): Analyzes descriptive metrics of the link.
        _analyze_influence_metrics(): Analyzes influence metrics of the link.
        _calculate_influence(metric_name, supply, supply_control): Calculates and stores influence metrics.
        _calculate_gini(metric_name, supply, supply_control): Calculates and stores Gini coefficients.
        _analyze_wealth_distribution(): Analyzes wealth distribution within the link.
        _analyze_labels(): Analyzes labels associated with the link.
        _calculate_max_influence_label(): Calculates the label with the maximum influence.
    """

    def __init__(self, link, dataFrame, sub_dataFrame, sub_dataFrame_control, token_lookup):
        self.link = link
        self.dataFrame = dataFrame
        self.sub_dataFrame = sub_dataFrame
        self.sub_dataFrame_control = sub_dataFrame_control
        self.token_lookup = token_lookup
        self.directional = False
        self.analysis_result = {}
        self.analysis_result_control = {}
        self.pvalues = {}

    def link_member_wallets(self):
        """
        Retrieve unique wallet addresses that hold ALL tokens in the provided link.

        Returns:
            list: A list of unique wallet addresses.
        """
        full_member_addresses = set(self.dataFrame[self.dataFrame.token_address == self.link[0]].address.unique())
        for token in self.link[1:]:
            token_addresses = set(self.dataFrame[self.dataFrame.token_address == token].address.unique())
            full_member_addresses &= token_addresses
            if not full_member_addresses:
                return []
        return list(full_member_addresses)

    def analyze_link(self):
        """
        Analyze the characteristics of a link of wallets within a given DataFrame.

        Returns:
            tuple: Contains the token names, analysis results, control analysis results, and p-values.
        """
        self._analyze_descriptive_metrics()
        self._analyze_influence_metrics()
        self._analyze_wealth_distribution()
        self._analyze_labels()
        return [self.token_lookup[i] for i in self.link], self.analysis_result, self.analysis_result_control, self.pvalues

    def _analyze_descriptive_metrics(self):
        """
        Analyzes descriptive metrics of the link.
        """
        self.analysis_result['size_link'] = self.sub_dataFrame.address.nunique()
        self.analysis_result_control['size_link'] = self.sub_dataFrame_control.address.nunique()

    def _analyze_influence_metrics(self):
        """
        Analyzes influence metrics of the link.
        """
        
        # total influence 
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        normalised_pct_supply = self.sub_dataFrame.pct_supply / no_token_communities_snapshot
        normalised_pct_supplyC = self.sub_dataFrame_control.pct_supply / no_token_communities_snapshot

        self._calculate_influence('total_influence', normalised_pct_supply, normalised_pct_supplyC)
        self._calculate_gini('gini_total_influence', normalised_pct_supply, normalised_pct_supplyC)

        # internal influence
        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.link)].pct_supply / len(self.link)
        normalised_pct_supply_internalC = self.sub_dataFrame_control[self.sub_dataFrame_control.token_address.isin(self.link)].pct_supply / len(self.link)

        self._calculate_influence('internal_influence', normalised_pct_supply_internal, normalised_pct_supply_internalC)
        self._calculate_gini('gini_internal_influence', normalised_pct_supply_internal, normalised_pct_supply_internalC)
        
        # external influence 
        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.link)].pct_supply / (no_token_communities_snapshot - len(self.link))
        normalised_pct_supply_externalC = self.sub_dataFrame_control[~self.sub_dataFrame_control.token_address.isin(self.link)].pct_supply / (no_token_communities_snapshot - len(self.link))

        self._calculate_influence('external_influence', normalised_pct_supply_external, normalised_pct_supply_externalC)
        self._calculate_gini('gini_external_influence', normalised_pct_supply_external, normalised_pct_supply_externalC)

    def _calculate_influence(self, metric_name, supply, supply_control):
        """
        Calculates and stores influence metrics.

        Args:
            metric_name (str): The name of the metric.
            supply (pd.Series): The supply data.
            supply_control (pd.Series): The control supply data.
        """
        influence_pval = permutation_test(supply, supply_control, alternative='greater')

        self.analysis_result[metric_name] = supply.sum()
        self.analysis_result_control[metric_name] = supply_control.sum()
        self.pvalues[metric_name] = influence_pval

    def _calculate_gini(self, metric_name, supply, supply_control):
        """
        Calculates and stores Gini coefficients.

        Args:
            metric_name (str): The name of the metric.
            supply (pd.Series): The supply data.
            supply_control (pd.Series): The control supply data.
        """
        gini_value = gini(supply)
        gini_value_control = gini(supply_control)
        gini_pval = permutation_test(supply, supply_control, alternative="lower")

        self.analysis_result[metric_name] = gini_value
        self.analysis_result_control[metric_name] = gini_value_control
        self.pvalues[metric_name] = gini_pval

    def _analyze_wealth_distribution(self):
        """
        Analyzes wealth distribution within the link.
        """
        wealth_link = self.sub_dataFrame['value_usd'].sum()
        total_wealth_pval = permutation_test(self.sub_dataFrame['value_usd'], self.sub_dataFrame_control['value_usd'], alternative='greater')

        self.analysis_result['wealth_link'] = wealth_link
        self.analysis_result_control['wealth_link'] = self.sub_dataFrame_control['value_usd'].sum()
        self.pvalues['wealth_link'] = total_wealth_pval

        gini_wealth_link = gini(self.sub_dataFrame['value_usd'])
        gini_wealth_linkC = gini(self.sub_dataFrame_control['value_usd'])
        gini_wealth_link_pval = permutation_test(self.sub_dataFrame['value_usd'], self.sub_dataFrame_control['value_usd'], alternative="lower")

        self.analysis_result['gini_wealth_link'] = gini_wealth_link
        self.analysis_result_control['gini_wealth_link'] = gini_wealth_linkC
        self.pvalues['gini_wealth_link'] = gini_wealth_link_pval

        median_wealth_level_link = self.sub_dataFrame.groupby('address')['value_usd'].sum().median()
        median_wealth_level_linkC = self.sub_dataFrame_control.groupby('address')['value_usd'].sum().median()
        median_wealth_level_link_pval = permutation_test(self.sub_dataFrame.groupby('address')['value_usd'].sum(), self.sub_dataFrame_control.groupby('address')['value_usd'].sum(), alternative='greater')

        self.analysis_result['median_wealth_level_link'] = median_wealth_level_link
        self.analysis_result_control['median_wealth_level_link'] = median_wealth_level_linkC
        self.pvalues['median_wealth_level_link'] = median_wealth_level_link_pval

        median_no_assets_link = self.sub_dataFrame.groupby('address')['token_address'].count().median()
        median_no_assets_linkC = self.sub_dataFrame_control.groupby('address')['token_address'].count().median()
        median_no_assets_link_pval = permutation_test(self.sub_dataFrame.groupby('address')['token_address'].count(), self.sub_dataFrame_control.groupby('address')['token_address'].count(), alternative='greater')

        self.analysis_result['median_no_assets_link'] = median_no_assets_link
        self.analysis_result_control['median_no_assets_link'] = median_no_assets_linkC
        self.pvalues['median_no_assets_link'] = median_no_assets_link_pval

    def _analyze_labels(self):
        """
        Analyzes labels associated with the link.

        """
        self.sub_dataFrame.fillna({'label': 'other_contracts'}, inplace=True)
        self.sub_dataFrame.fillna({'label': 'other_contracts'}, inplace=True)

        self._calculate_max_influence_label_distribution()

    def _calculate_max_influence_label_distribution(self):
        """
        Calculates the label with the maximum influence distribution.
        """

        self.analysis_result['max_influence_label_distribution'] = dict(self.sub_dataFrame.groupby(['label'])['pct_supply'].sum() / self.dataFrame.token_address.nunique())
        self.analysis_result_control['max_influence_label_distribution'] = dict(self.sub_dataFrame_control.groupby(['label'])['pct_supply'].sum() / self.dataFrame.token_address.nunique())

def analyze_link(sub_dataFrame, sub_dataFrame_control, dataFrame, link, token_lookup):
    """
    Function to analyze a link of wallets within given DataFrames.

    Args:
        sub_dataFrame (pd.DataFrame): Subset of the main DataFrame for the link.
        sub_dataFrame_control (pd.DataFrame): Control subset of the DataFrame.
        dataFrame (pd.DataFrame): The main DataFrame containing token data.
        link (list): List of token addresses in the link.
        token_lookup (dict): Dictionary for token address to token name mapping.

    Returns:
        tuple: Contains the token names, analysis results, control analysis results, and p-values.
    """
    analyzer = LinkAnalysis(link, dataFrame, sub_dataFrame, sub_dataFrame_control, token_lookup)
    return analyzer.analyze_link()
