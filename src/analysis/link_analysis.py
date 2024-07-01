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
        if self.directional==False: 
        

            # descriptive
            self._calculate_clique_size()
            self._calculate_token_holding_count()
            self._analyze_labels()

            # INFLUENCE 
            # total influence 
            self._calculate_total_influence()
            self._calculate_gini_of_total_influence()
            self._calculate_median_influence()

            # internal influence 
            self._calculate_internal_influence()
            self._calculate_gini_of_internal_influence()

            # external influence 
            self._calculate_external_influence()
            self._calculate_gini_of_external_influence()


            # WEALTH
            # total wealth
            self._calculate_total_link_wealth()
            self._calculate_total_link_wealth_gini()
            self._calculate_median_wealth()

            # total wealth of Link Token (equivalent of internal influence)
            

            # total wealth of Non-Link Token (equivalent of external influence)
            

            # LABELS
            self._analyze_labels()


        elif self.directional==True: 

            # descriptive
            self._calculate_clique_size()
            self._calculate_token_holding_count()
            self._analyze_labels()

            # INFLUENCE 
            # total influence 
            self._calculate_total_influence_directional()

            # WEALTH
            # total wealth
            self._calculate_total_link_wealth()
            self._calculate_total_link_wealth_gini()
            self._calculate_median_wealth()

           
            return [self.token_lookup[i] for i in self.link], self.analysis_result, self.analysis_result_control, self.pvalues
        

    ###############################
    ### Descriptive Metrice #######
    ###############################


    def _calculate_clique_size(self):
        """
        Analyzes descriptive metrics of the link.
        """
        self.analysis_result['size_link'] = self.sub_dataFrame.address.nunique()
        self.analysis_result_control['size_link'] = self.sub_dataFrame_control.address.nunique()   


    def _calculate_token_holding_count(self):
        """
        Analyzes median number of asset held by link-defining addresses within a link.

        """
        median_no_assets_link = self.sub_dataFrame.groupby('address')['token_address'].count()
        median_no_assets_linkC = self.sub_dataFrame_control.groupby('address')['token_address'].count()
        median_no_assets_link_pval = permutation_test(median_no_assets_link,median_no_assets_linkC, method='median', alternative='greater')

        self.analysis_result['median_no_assets_link'] = median_no_assets_link.median()
        self.analysis_result_control['median_no_assets_link'] = median_no_assets_linkC.median()
        self.pvalues['median_no_assets_link'] = median_no_assets_link_pval

    def _analyze_labels(self):
        """
        Analyzes labels associated with the link.

        """
        self.sub_dataFrame.fillna({'label': 'other_contracts'}, inplace=True)
        self.sub_dataFrame.fillna({'label': 'other_contracts'}, inplace=True)

        self.analysis_result['max_influence_label_distribution'] = dict(self.sub_dataFrame.groupby(['label'])['pct_supply'].sum() / self.dataFrame.token_address.nunique())
        self.analysis_result_control['max_influence_label_distribution'] = dict(self.sub_dataFrame_control.groupby(['label'])['pct_supply'].sum() / self.dataFrame.token_address.nunique())




    ###############################
    ##### Analysis Metrice ########
    ###############################

    def _calculate_total_influence(self):
        """
        Analyzes total influence of the link.
        """

        # total influence 
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        # total influence normaisation
        normalised_pct_supply = self.sub_dataFrame.pct_supply / no_token_communities_snapshot
        normalised_pct_supplyC = self.sub_dataFrame_control.pct_supply / no_token_communities_snapshot
        normalised_pct_link_pval = permutation_test(normalised_pct_supply, normalised_pct_supplyC, method='mean', alternative='greater')

        self.analysis_result['total_influence'] = normalised_pct_supply.sum()
        self.analysis_result_control['total_influence'] = normalised_pct_supplyC.sum()
        self.pvalues['total_influence'] = normalised_pct_link_pval

    def _calculate_gini_of_total_influence(self):
        """
        Analyzes total influence of the link.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        # gini total influence 
        normalised_pct_supply = self.sub_dataFrame.pct_supply / no_token_communities_snapshot
        normalised_pct_supplyC = self.sub_dataFrame_control.pct_supply / no_token_communities_snapshot
        normalised_pct_link_pval = permutation_test(normalised_pct_supply, normalised_pct_supplyC, method='gini', alternative='lower')

        self.analysis_result['gini_total_influence'] = gini(normalised_pct_supply)
        self.analysis_result_control['gini_total_influence'] = gini(normalised_pct_supplyC)
        self.pvalues['gini_total_influence'] = normalised_pct_link_pval

    def _calculate_median_influence(self):
        """
        Analyzes influence metrics of the link.
        """
        # median total influence 
        median_influence_level_link = self.sub_dataFrame.groupby('address')['pct_supply']
        median_influence_level_linkC = self.sub_dataFrame_control.groupby('address')['pct_supply']
        median_influence_level_link_pval = permutation_test(median_influence_level_link, median_influence_level_linkC, method='median', alternative='greater')

        self.analysis_result['median_wealth_level_link'] = median_influence_level_link.median()
        self.analysis_result_control['median_wealth_level_link'] = median_influence_level_linkC.median()
        self.pvalues['median_wealth_level_link'] = median_influence_level_link_pval

    def _calculate_internal_influence(self):
        """
        Analyzes total influence of the link.
        """
        # internal influence
        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.link)].pct_supply / len(self.link)
        normalised_pct_supply_internalC = self.sub_dataFrame_control[self.sub_dataFrame_control.token_address.isin(self.link)].pct_supply / len(self.link)
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='mean', alternative='greater')

        self.analysis_result['internal_influence'] = normalised_pct_supply_internal.sum()
        self.analysis_result_control['internal_influence'] = normalised_pct_supply_internalC.sum()
        self.pvalues['internal_influence'] = normalised_pct_link_pval

    def _calculate_gini_of_internal_influence(self):
        """
        Analyzes total influence of the link.
        """

        # gini total influence 
        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.link)].pct_supply / len(self.link)
        normalised_pct_supply_internalC = self.sub_dataFrame_control[self.sub_dataFrame_control.token_address.isin(self.link)].pct_supply / len(self.link)
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='gini', alternative='lower')

        self.analysis_result['gini_total_influence'] = gini(normalised_pct_supply_internal)
        self.analysis_result_control['gini_total_influence'] = gini(normalised_pct_supply_internalC)
        self.pvalues['gini_total_influence'] = normalised_pct_link_pval

    def _calculate_external_influence(self):
        """
        Analyzes total influence of the link.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        # internal influence
        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.link)].pct_supply / (no_token_communities_snapshot - len(self.link))
        normalised_pct_supply_externalC = self.sub_dataFrame_control[~self.sub_dataFrame_control.token_address.isin(self.link)].pct_supply / (no_token_communities_snapshot - len(self.link))
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='mean', alternative='greater')

        self.analysis_result['internal_influence'] = normalised_pct_supply_external.sum()
        self.analysis_result_control['internal_influence'] = normalised_pct_supply_externalC.sum()
        self.pvalues['internal_influence'] = normalised_pct_link_pval

    def _calculate_gini_of_external_influence(self):
        """
        Analyzes total influence of the link.
        """
        
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        # gini total influence 
        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.link)].pct_supply / (no_token_communities_snapshot - len(self.link))
        normalised_pct_supply_externalC = self.sub_dataFrame_control[~self.sub_dataFrame_control.token_address.isin(self.link)].pct_supply / (no_token_communities_snapshot - len(self.link))
        normalised_pct_link_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='gini', alternative='lower')

        self.analysis_result['gini_total_influence'] = gini(normalised_pct_supply_external)
        self.analysis_result_control['gini_total_influence'] = gini(normalised_pct_supply_externalC)
        self.pvalues['gini_total_influence'] = normalised_pct_link_pval            

    def _calculate_total_influence_directional(self):
        """
        Analyzes total influence of the link.
        """
        # total influence normaisation
        normalised_pct_supply = self.sub_dataFrame.pct_supply 
        normalised_pct_supplyC = self.sub_dataFrame_control.pct_supply
        normalised_pct_link_pval = permutation_test(normalised_pct_supply, normalised_pct_supplyC, method='mean', alternative='greater')

        self.analysis_result['total_influence'] = normalised_pct_supply.sum()
        self.analysis_result_control['total_influence'] = normalised_pct_supplyC.sum()
        self.pvalues['total_influence'] = normalised_pct_link_pval

    ###########################
    ##### Wealth Metrics ######
    ###########################

    def _calculate_total_link_wealth(self):
        """
        Analyzes wealth distribution within the link.
        """
        wealth_level_link = self.sub_dataFrame.groupby('address')['value_usd']
        wealth_level_linkC = self.sub_dataFrame_control.groupby('address')['value_usd']
        normalised_pct_link_pval = permutation_test(wealth_level_link, wealth_level_linkC, method='mean', alternative='greater')

        self.analysis_result['total_wealth_level_link'] = wealth_level_link.sum()
        self.analysis_result_control['total_wealth_level_link'] = wealth_level_linkC.sum()
        self.pvalues['total_wealth_level_link'] = normalised_pct_link_pval

    def _calculate_total_link_wealth_gini(self): 
        gini_wealth_link = self.sub_dataFrame.groupby('address')['value_usd']
        gini_wealth_linkC = self.sub_dataFrame_control.groupby('address')['value_usd']
        normalised_pct_link_pval = permutation_test(gini_wealth_link, gini_wealth_linkC, method='gini', alternative='greater')

        self.analysis_result['gini_wealth_link'] = gini_wealth_link
        self.analysis_result_control['gini_wealth_link'] = gini_wealth_linkC
        self.pvalues['gini_wealth_link'] = normalised_pct_link_pval

    def _calculate_median_wealth(self):
        """
        Analyzes influence metrics of the link.
        """

        # median total influence 
        median_wealth_level_link = self.sub_dataFrame.groupby('address')['value_usd']
        median_wealth_level_linkC = self.sub_dataFrame_control.groupby('address')['value_usd']
        median_wealth_level_link_pval = permutation_test(median_wealth_level_link, median_wealth_level_linkC, method='median', alternative='greater')

        self.analysis_result['median_wealth_level_link'] = median_wealth_level_link.median()
        self.analysis_result_control['median_wealth_level_link'] = median_wealth_level_linkC.median()
        self.pvalues['median_wealth_level_link'] = median_wealth_level_link_pval
