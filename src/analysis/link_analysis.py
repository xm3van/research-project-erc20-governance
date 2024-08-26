import pandas as pd
import numpy as np
import pickle
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
        # Manually define all possible metric names
        metric_names = [
            'size',
            'median_number_assets',
            'max_influence_label_distribution',
            'total_influence',
            'gini_total_influence',
            'median_influence',
            'internal_influence',
            'gini_internal_influence',
            'external_influence',
            'gini_external_influence',
            'total_influence_directional',
            'total_wealth',
            'gini_total_wealth',
            'median_wealth',
            'internal_wealth',
            'gini_internal_wealth',
            'external_wealth',
            'gini_external_wealth'
        ]
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
    Class for conducting analysis on link data, focusing on metrics related to influence and wealth distribution.

    Attributes:
        link (list): A list of token addresses in the link.
        dataFrame (pd.DataFrame): The primary DataFrame containing token data.
        sub_dataFrame (pd.DataFrame): A subset of the main DataFrame specific to the link.
        sub_dataFrame_sample_population (pd.DataFrame): A control subset of the DataFrame for comparison.
        token_lookup (dict): A dictionary for mapping token addresses to token names.
        analysis_result (dict): A dictionary to store the results of the analysis.
        analysis_result_sample_population (dict): A dictionary to store the results of the sample population analysis.
        pvalues (dict): A dictionary to store p-values of the statistical tests.

    Methods:
        link_member_wallets(): Retrieves unique wallet addresses that hold all tokens in the provided link.
        analyze_link(): Analyzes the characteristics of the link.
        _calculate_link_size(): Computes the size of the link.
        _calculate_token_holding_count(): Computes the median number of assets held by link-defining addresses.
        _analyze_labels(): Analyzes the labels associated with the link.
        _calculate_total_influence(): Computes the total influence of the link.
        _calculate_gini_of_total_influence(): Computes the Gini coefficient for the total influence.
        _calculate_median_influence(): Computes the median influence level within the link.
        _calculate_internal_influence(): Computes the internal influence of the link.
        _calculate_gini_of_internal_influence(): Computes the Gini coefficient for the internal influence.
        _calculate_external_influence(): Computes the external influence of the link.
        _calculate_gini_of_external_influence(): Computes the Gini coefficient for the external influence.
        _calculate_total_influence_directional(): Computes the directional total influence of the link.
        _calculate_total_link_wealth(): Computes the total wealth within the link.
        _calculate_gini_of_total_link_wealth(): Computes the Gini coefficient for the total wealth within the link.
        _calculate_median_wealth(): Computes the median wealth level within the link.
        _calculate_internal_wealth(): Computes the internal wealth of the link.
        _calculate_gini_of_internal_wealth(): Computes the Gini coefficient for the internal wealth.
        _calculate_external_wealth(): Computes the external wealth of the link.
        _calculate_gini_of_external_wealth(): Computes the Gini coefficient for the external wealth.
    """

    def __init__(self, link, dataFrame, sub_dataFrame, sub_dataFrame_sample_population, token_lookup):
        self.link = link
        self.dataFrame = dataFrame
        self.sub_dataFrame = sub_dataFrame
        self.sub_dataFrame_sample_population = sub_dataFrame_sample_population
        self.token_lookup = token_lookup
        self.directional = False
        self.analysis_result = {}
        self.analysis_result_sample_population = {}
        self.pvalues = {}

    def link_member_wallets(self):
        """
        Retrieves unique wallet addresses that hold all tokens in the provided link.

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
        Analyzes the characteristics of the link within a given DataFrame.

        Returns:
            tuple: Contains the token names, analysis results, sample population analysis results, and p-values.
        """
        if self.directional == False:
            # Descriptive metrics
            self._calculate_size()
            self._calculate_median_token_holding_count()
            self._analyze_labels()

            # Influence metrics
            ## Total Influence
            self._calculate_total_influence()
            self._calculate_gini_of_total_influence()
            self._calculate_median_influence()
            ## internal influence 
            self._calculate_internal_influence()
            self._calculate_gini_of_internal_influence()
            ## external influence 
            self._calculate_external_influence()
            self._calculate_gini_of_external_influence()

            # Wealth metrics
            ## Total wealth
            self._calculate_total_wealth()
            self._calculate_gini_of_total_wealth()
            self._calculate_median_wealth()

            ## Internal wealth
            self._calculate_internal_wealth()
            self._calculate_gini_of_internal_wealth()

            ## External Wealth
            self._calculate_external_wealth()
            self._calculate_gini_of_external_wealth()

            # Labels
            self._analyze_labels()

        if self.directional == True:
            # Descriptive metrics
            self._calculate_size()
            self._calculate_median_token_holding_count()
            self._analyze_labels()

            # Influence metrics
            self._calculate_total_influence_directional()
            self._calculate_gini_of_total_influence()

            # Wealth metrics
            self._calculate_total_wealth()
            self._calculate_gini_of_total_wealth()
            self._calculate_median_wealth()

        return [self.token_lookup[i] for i in self.link], self.analysis_result, self.analysis_result_sample_population, self.pvalues

    ###############################
    ### Descriptive Metrics #######
    ###############################

    def _calculate_size(self):
        """
        Computes the size of the link.
        """
        self.analysis_result['size'] = self.sub_dataFrame.address.nunique()
        self.analysis_result_sample_population['size'] = self.sub_dataFrame_sample_population.address.nunique()

    def _calculate_median_token_holding_count(self):
        """
        Computes the median number of assets held by link-defining addresses within a link.
        """
        median_number_assets_link = self.sub_dataFrame.groupby('address')['token_address'].count()
        median_number_assets_linkC = self.sub_dataFrame_sample_population.groupby('address')['token_address'].count()
        median_number_assets_link_pval = permutation_test(median_number_assets_link, median_number_assets_linkC, method='median', alternative='greater')

        self.analysis_result['median_number_assets'] = median_number_assets_link.median()
        self.analysis_result_sample_population['median_number_assets'] = median_number_assets_linkC.median()
        self.pvalues['median_number_assets'] = median_number_assets_link_pval

    def _analyze_labels(self):
        """
        Analyzes the labels associated with the link.
        """
        self.sub_dataFrame.fillna({'label': 'other_contracts'}, inplace=True)
        self.sub_dataFrame_sample_population.fillna({'label': 'other_contracts'}, inplace=True)

        self.analysis_result['max_influence_label_distribution'] = dict(self.sub_dataFrame.groupby(['label'])['pct_supply'].sum() / self.dataFrame.token_address.nunique())
        self.analysis_result_sample_population['max_influence_label_distribution'] = dict(self.sub_dataFrame_sample_population.groupby(['label'])['pct_supply'].sum() / self.dataFrame.token_address.nunique())

    ###############################
    ##### Influence Metrics #######
    ###############################

    def _calculate_total_influence(self):
        """
        Computes the total influence of the link.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        normalised_pct_supply = self.sub_dataFrame.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        normalised_pct_supplyC = self.sub_dataFrame_sample_population.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        normalised_pct_link_pval = permutation_test(normalised_pct_supply, normalised_pct_supplyC, method='mean', alternative='greater')

        self.analysis_result['total_influence'] = normalised_pct_supply.sum()
        self.analysis_result_sample_population['total_influence'] = normalised_pct_supplyC.sum()
        self.pvalues['total_influence'] = normalised_pct_link_pval

    def _calculate_gini_of_total_influence(self):
        """
        Computes the Gini coefficient for the total influence.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        normalised_pct_supply = self.sub_dataFrame.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        normalised_pct_supplyC = self.sub_dataFrame_sample_population.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        normalised_pct_link_pval = permutation_test(normalised_pct_supply, normalised_pct_supplyC, method='gini', alternative='lower')

        self.analysis_result['gini_total_influence'] = gini(normalised_pct_supply)
        self.analysis_result_sample_population['gini_total_influence'] = gini(normalised_pct_supplyC)
        self.pvalues['gini_total_influence'] = normalised_pct_link_pval

    def _calculate_median_influence(self):
        """
        Computes the median influence level within the link.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        median_influence_link = self.sub_dataFrame.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        median_influence_linkC = self.sub_dataFrame_sample_population.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        median_influence_link_pval = permutation_test(median_influence_link, median_influence_linkC, method='median', alternative='greater')

        self.analysis_result['median_influence'] = median_influence_link.median()
        self.analysis_result_sample_population['median_influence'] = median_influence_linkC.median()
        self.pvalues['median_influence'] = median_influence_link_pval

    def _calculate_internal_influence(self):
        """
        Computes the internal influence of the link.
        """
        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.link)].groupby('address')['pct_supply'].sum() / len(self.link)
        normalised_pct_supply_internalC = self.sub_dataFrame_sample_population[self.sub_dataFrame_sample_population.token_address.isin(self.link)].groupby('address')['pct_supply'].sum() / len(self.link)
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='mean', alternative='greater')

        self.analysis_result['internal_influence'] = normalised_pct_supply_internal.sum()
        self.analysis_result_sample_population['internal_influence'] = normalised_pct_supply_internalC.sum()
        self.pvalues['internal_influence'] = normalised_pct_link_pval

    def _calculate_gini_of_internal_influence(self):
        """
        Computes the Gini coefficient for the internal influence.
        """
        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.link)].groupby('address')['pct_supply'].sum() / len(self.link)
        normalised_pct_supply_internalC = self.sub_dataFrame_sample_population[self.sub_dataFrame_sample_population.token_address.isin(self.link)].groupby('address')['pct_supply'].sum() / len(self.link)
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='gini', alternative='lower')

        self.analysis_result['gini_internal_influence'] = gini(normalised_pct_supply_internal)
        self.analysis_result_sample_population['gini_internal_influence'] = gini(normalised_pct_supply_internalC)
        self.pvalues['gini_internal_influence'] = normalised_pct_link_pval

    def _calculate_external_influence(self):
        """
        Computes the external influence of the link.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.link)].groupby('address')['pct_supply'].sum() / (no_token_communities_snapshot - len(self.link))
        normalised_pct_supply_externalC = self.sub_dataFrame_sample_population[~self.sub_dataFrame_sample_population.token_address.isin(self.link)].groupby('address')['pct_supply'].sum() / (no_token_communities_snapshot - len(self.link))
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='mean', alternative='greater')

        self.analysis_result['external_influence'] = normalised_pct_supply_external.sum()
        self.analysis_result_sample_population['external_influence'] = normalised_pct_supply_externalC.sum()
        self.pvalues['external_influence'] = normalised_pct_link_pval

    def _calculate_gini_of_external_influence(self):
        """
        Computes the Gini coefficient for the external influence.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.link)].groupby('address')['pct_supply'].sum() / (no_token_communities_snapshot - len(self.link))
        normalised_pct_supply_externalC = self.sub_dataFrame_sample_population[~self.sub_dataFrame_sample_population.token_address.isin(self.link)].groupby('address')['pct_supply'].sum() / (no_token_communities_snapshot - len(self.link))
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='gini', alternative='lower')

        self.analysis_result['gini_external_influence'] = gini(normalised_pct_supply_external)
        self.analysis_result_sample_population['gini_external_influence'] = gini(normalised_pct_supply_externalC)
        self.pvalues['gini_external_influence'] = normalised_pct_link_pval

    def _calculate_total_influence_directional(self):
        """
        Computes the directional total influence of the link.
        """
        normalised_pct_supply = self.sub_dataFrame.groupby('address')['pct_supply'].sum()
        normalised_pct_supplyC = self.sub_dataFrame_sample_population.groupby('address')['pct_supply'].sum()
        normalised_pct_link_pval = permutation_test(normalised_pct_supply, normalised_pct_supplyC, method='mean', alternative='greater')

        self.analysis_result['total_influence_directional'] = normalised_pct_supply.sum()
        self.analysis_result_sample_population['total_influence_directional'] = normalised_pct_supplyC.sum()
        self.pvalues['total_influence_directional'] = normalised_pct_link_pval

    ###########################
    ##### Wealth Metrics ######
    ###########################

    def _calculate_total_wealth(self):
        """
        Computes the total wealth within the link.
        """
        wealth_link = self.sub_dataFrame.groupby('address')['value_usd'].sum()
        wealth_linkC = self.sub_dataFrame_sample_population.groupby('address')['value_usd'].sum()
        normalised_pct_link_pval = permutation_test(wealth_link, wealth_linkC, method='mean', alternative='greater')

        self.analysis_result['total_wealth'] = wealth_link.sum()
        self.analysis_result_sample_population['total_wealth'] = wealth_linkC.sum()
        self.pvalues['total_wealth'] = normalised_pct_link_pval

    def _calculate_gini_of_total_wealth(self):
        """
        Computes the Gini coefficient for the total wealth within the link.
        """
        gini_wealth_link = self.sub_dataFrame.groupby('address')['value_usd'].sum()
        gini_wealth_linkC = self.sub_dataFrame_sample_population.groupby('address')['value_usd'].sum()
        normalised_pct_link_pval = permutation_test(gini_wealth_link, gini_wealth_linkC, method='gini', alternative='greater')

        self.analysis_result['gini_total_wealth'] = gini(gini_wealth_link)
        self.analysis_result_sample_population['gini_total_wealth'] = gini(gini_wealth_linkC)
        self.pvalues['gini_total_wealth'] = normalised_pct_link_pval

    def _calculate_median_wealth(self):
        """
        Computes the median wealth  within the link.
        """
        median_wealth_link = self.sub_dataFrame.groupby('address')['value_usd'].sum()
        median_wealth_linkC = self.sub_dataFrame_sample_population.groupby('address')['value_usd'].sum()
        median_wealth_link_pval = permutation_test(median_wealth_link, median_wealth_linkC, method='median', alternative='greater')

        self.analysis_result['median_wealth'] = median_wealth_link.median()
        self.analysis_result_sample_population['median_wealth'] = median_wealth_linkC.median()
        self.pvalues['median_wealth'] = median_wealth_link_pval

    def _calculate_internal_wealth(self):
        """
        Computes the internal wealth of the link (value of tokens that are part of the link).
        """
        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.link)].groupby('address')['value_usd'].sum()
        normalised_pct_supply_internalC = self.sub_dataFrame_sample_population[self.sub_dataFrame_sample_population.token_address.isin(self.link)].groupby('address')['value_usd'].sum()
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='mean', alternative='greater')

        self.analysis_result['internal_wealth'] = normalised_pct_supply_internal.sum()
        self.analysis_result_sample_population['internal_wealth'] = normalised_pct_supply_internalC.sum()
        self.pvalues['internal_wealth'] = normalised_pct_link_pval

    def _calculate_gini_of_internal_wealth(self):
        """
        Computes the Gini coefficient for the internal wealth.
        """
        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.link)].groupby('address')['value_usd'].sum()
        normalised_pct_supply_internalC = self.sub_dataFrame_sample_population[self.sub_dataFrame_sample_population.token_address.isin(self.link)].groupby('address')['value_usd'].sum()
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='gini', alternative='lower')

        self.analysis_result['gini_internal_wealth'] = gini(normalised_pct_supply_internal)
        self.analysis_result_sample_population['gini_internal_wealth'] = gini(normalised_pct_supply_internalC)
        self.pvalues['gini_internal_wealth'] = normalised_pct_link_pval

    def _calculate_external_wealth(self):
        """
        Computes the external wealth of the link (wealth in tokens not part of the link but part of the sample).
        """
        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.link)].groupby('address')['value_usd'].sum()
        normalised_pct_supply_externalC = self.sub_dataFrame_sample_population[~self.sub_dataFrame_sample_population.token_address.isin(self.link)].groupby('address')['value_usd'].sum()
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='mean', alternative='greater')

        self.analysis_result['external_wealth'] = normalised_pct_supply_external.sum()
        self.analysis_result_sample_population['external_wealth'] = normalised_pct_supply_externalC.sum()
        self.pvalues['external_wealth'] = normalised_pct_link_pval

    def _calculate_gini_of_external_wealth(self):
        """
        Computes the Gini coefficient for the external wealth.
        """
        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.link)].groupby('address')['value_usd'].sum()
        normalised_pct_supply_externalC = self.sub_dataFrame_sample_population[~self.sub_dataFrame_sample_population.token_address.isin(self.link)].groupby('address')['value_usd'].sum()
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='gini', alternative='lower')

        self.analysis_result['gini_external_wealth'] = gini(normalised_pct_supply_external)
        self.analysis_result_sample_population['gini_external_wealth'] = gini(normalised_pct_supply_externalC)
        self.pvalues['gini_external_wealth'] = normalised_pct_link_pval
