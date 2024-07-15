import pandas as pd
import numpy as np
import pickle
from os.path import join
from itertools import combinations
from src.utilities.metrics_and_tests import *
from src.utilities.utils import *

class CliquesData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.cliques = self.load_cliques()
        self.metric_names = self.get_metric_names()

    def load_cliques(self):
        with open(self.data_path, 'rb') as handle:
            cliques = pickle.load(handle)
        return cliques

    def get_metric_names(self):
        example_key = next(iter(self.cliques['strong_estimate']['sample']))
        metrics_example = self.cliques['strong_estimate']['sample'][example_key]
        first_clique_key = next(iter(metrics_example.keys()))
        metric_names = list(metrics_example[first_clique_key].keys())
        return metric_names

    def get_metric_data(self, bound, group, metric_name):
        data = []

        for date, snapshot_data in self.cliques[bound][group].items():
            for clique_name, metrics in snapshot_data.items():
                try:
                    metric_value = metrics[metric_name]
                except KeyError:
                    metric_value = np.nan
                data.append({'Date': date, 'Clique Name': clique_name, metric_name: metric_value})

        df_metric = pd.DataFrame(data)
        df_metric = df_metric.pivot(index='Clique Name', columns='Date', values=metric_name)

        return df_metric


class CliqueAnalysis:
    """
    Class for conducting analysis on clique data, focusing on metrics related to influence and wealth distribution.

    Attributes:
        clique (list): A list of token addresses in the clique.
        dataFrame (pd.DataFrame): The primary DataFrame containing token data.
        sub_dataFrame (pd.DataFrame): A subset of the main DataFrame specific to the clique.
        sub_dataFrame_sample_population (pd.DataFrame): A control subset of the DataFrame for comparison.
        token_lookup (dict): A dictionary for mapping token addresses to token names.
        analysis_result (dict): A dictionary to store the results of the analysis.
        analysis_result_sample_population (dict): A dictionary to store the results of the sample population analysis.
        pvalues (dict): A dictionary to store p-values of the statistical tests.

    Methods:
        clique_member_wallets(): Retrieves unique wallet addresses that hold all tokens in the provided clique.
        analyze_clique(): Analyzes the characteristics of the clique.
        _calculate_clique_size(): Computes the size of the clique.
        _calculate_token_holding_count(): Computes the median number of assets held by clique-defining addresses.
        _analyze_labels(): Analyzes the labels associated with the clique.
        _calculate_total_influence(): Computes the total influence of the clique.
        _calculate_gini_of_total_influence(): Computes the Gini coefficient for the total influence.
        _calculate_median_influence(): Computes the median influence  within the clique.
        _calculate_internal_influence(): Computes the internal influence of the clique.
        _calculate_gini_of_internal_influence(): Computes the Gini coefficient for the internal influence.
        _calculate_external_influence(): Computes the external influence of the clique.
        _calculate_gini_of_external_influence(): Computes the Gini coefficient for the external influence.
        _calculate_total_influence_directional(): Computes the directional total influence of the clique.
        _calculate_total_clique_wealth(): Computes the total wealth within the clique.
        _calculate_gini_of_total_clique_wealth(): Computes the Gini coefficient for the total wealth within the clique.
        _calculate_median_wealth(): Computes the median wealth  within the clique.
        _calculate_internal_wealth(): Computes the internal wealth of the clique.
        _calculate_gini_of_internal_wealth(): Computes the Gini coefficient for the internal wealth.
        _calculate_external_wealth(): Computes the external wealth of the clique.
        _calculate_gini_of_external_wealth(): Computes the Gini coefficient for the external wealth.
    """

    def __init__(self, clique, dataFrame, sub_dataFrame, sub_dataFrame_sample_population, token_lookup):
        self.clique = clique
        self.dataFrame = dataFrame
        self.sub_dataFrame = sub_dataFrame
        self.sub_dataFrame_sample_population = sub_dataFrame_sample_population
        self.token_lookup = token_lookup
        self.directional = False
        self.analysis_result = {}
        self.analysis_result_sample_population = {}
        self.pvalues = {}

    def clique_member_wallets_weak(self):
        clique_members = set()
        for c in combinations(self.clique, 2):
            set1_addresses = set(self.dataFrame[self.dataFrame.token_address == c[0]].address.unique())
            set2_addresses = set(self.dataFrame[self.dataFrame.token_address == c[1]].address.unique())
            clique_members.update(set1_addresses & set2_addresses)
        return list(clique_members)

    def clique_member_wallets_strong(self):
        full_member_addresses = set(self.dataFrame[self.dataFrame.token_address == self.clique[0]].address.unique())
        for token in self.clique[1:]:
            token_addresses = set(self.dataFrame[self.dataFrame.token_address == token].address.unique())
            full_member_addresses &= token_addresses
            if not full_member_addresses:
                return []
        return list(full_member_addresses)

    def analyze_clique(self):
        """
        Analyzes the characteristics of the clique within a given DataFrame.

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

        elif self.directional == True:
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

        return [self.token_lookup[i] for i in self.clique], self.analysis_result, self.analysis_result_sample_population, self.pvalues

    ###############################
    ### Descriptive Metrics #######
    ###############################

    def _calculate_size(self):
        """
        Computes the size of the clique.
        """
        self.analysis_result['size'] = self.sub_dataFrame.address.nunique()
        self.analysis_result_sample_population['size'] = self.sub_dataFrame_sample_population.address.nunique()

    def _calculate_median_token_holding_count(self):
        """
        Computes the median number of assets held by clique-defining addresses within a clique.
        """
        median_number_assets_clique = self.sub_dataFrame.groupby('address')['token_address'].count()
        median_number_assets_cliqueC = self.sub_dataFrame_sample_population.groupby('address')['token_address'].count()
        median_number_assets_clique_pval = permutation_test(median_number_assets_clique, median_number_assets_cliqueC, method='median', alternative='greater')

        self.analysis_result['median_number_assets'] = median_number_assets_clique.median()
        self.analysis_result_sample_population['median_number_assets'] = median_number_assets_cliqueC.median()
        self.pvalues['median_number_assets'] = median_number_assets_clique_pval

    def _analyze_labels(self):
        """
        Analyzes the labels associated with the clique.
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
        Computes the total influence of the clique.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        normalised_pct_supply = self.sub_dataFrame.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        normalised_pct_supplyC = self.sub_dataFrame_sample_population.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        normalised_pct_clique_pval = permutation_test(normalised_pct_supply, normalised_pct_supplyC, method='mean', alternative='greater')

        self.analysis_result['total_influence'] = normalised_pct_supply.sum()
        self.analysis_result_sample_population['total_influence'] = normalised_pct_supplyC.sum()
        self.pvalues['total_influence'] = normalised_pct_clique_pval

    def _calculate_gini_of_total_influence(self):
        """
        Computes the Gini coefficient for the total influence.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        normalised_pct_supply = self.sub_dataFrame.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        normalised_pct_supplyC = self.sub_dataFrame_sample_population.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        normalised_pct_clique_pval = permutation_test(normalised_pct_supply, normalised_pct_supplyC, method='gini', alternative='lower')

        self.analysis_result['gini_total_influence'] = gini(normalised_pct_supply)
        self.analysis_result_sample_population['gini_total_influence'] = gini(normalised_pct_supplyC)
        self.pvalues['gini_total_influence'] = normalised_pct_clique_pval

    def _calculate_median_influence(self):
        """
        Computes the median influence  within the clique.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        median_influence_clique = self.sub_dataFrame.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        median_influence_cliqueC = self.sub_dataFrame_sample_population.groupby('address')['pct_supply'].sum() / no_token_communities_snapshot
        median_influence_clique_pval = permutation_test(median_influence_clique, median_influence_cliqueC, method='median', alternative='greater')

        self.analysis_result['median_influence'] = median_influence_clique.median()
        self.analysis_result_sample_population['median_influence'] = median_influence_cliqueC.median()
        self.pvalues['median_influence'] = median_influence_clique_pval

    def _calculate_internal_influence(self):
        """
        Computes the internal influence of the clique.
        """
        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.clique)].groupby('address')['pct_supply'].sum() / len(self.clique)
        normalised_pct_supply_internalC = self.sub_dataFrame_sample_population[self.sub_dataFrame_sample_population.token_address.isin(self.clique)].groupby('address')['pct_supply'].sum() / len(self.clique)
        normalised_pct_clique_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='mean', alternative='greater')

        self.analysis_result['internal_influence'] = normalised_pct_supply_internal.sum()
        self.analysis_result_sample_population['internal_influence'] = normalised_pct_supply_internalC.sum()
        self.pvalues['internal_influence'] = normalised_pct_clique_pval

    def _calculate_gini_of_internal_influence(self):
        """
        Computes the Gini coefficient for the internal influence.
        """
        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.clique)].groupby('address')['pct_supply'].sum() / len(self.clique)
        normalised_pct_supply_internalC = self.sub_dataFrame_sample_population[self.sub_dataFrame_sample_population.token_address.isin(self.clique)].groupby('address')['pct_supply'].sum() / len(self.clique)
        normalised_pct_clique_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='gini', alternative='lower')

        self.analysis_result['gini_internal_influence'] = gini(normalised_pct_supply_internal)
        self.analysis_result_sample_population['gini_internal_influence'] = gini(normalised_pct_supply_internalC)
        self.pvalues['gini_internal_influence'] = normalised_pct_clique_pval

    def _calculate_external_influence(self):
        """
        Computes the external influence of the clique.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.clique)].groupby('address')['pct_supply'].sum() / (no_token_communities_snapshot - len(self.clique))
        normalised_pct_supply_externalC = self.sub_dataFrame_sample_population[~self.sub_dataFrame_sample_population.token_address.isin(self.clique)].groupby('address')['pct_supply'].sum() / (no_token_communities_snapshot - len(self.clique))
        normalised_pct_clique_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='mean', alternative='greater')

        self.analysis_result['external_influence'] = normalised_pct_supply_external.sum()
        self.analysis_result_sample_population['external_influence'] = normalised_pct_supply_externalC.sum()
        self.pvalues['external_influence'] = normalised_pct_clique_pval

    def _calculate_gini_of_external_influence(self):
        """
        Computes the Gini coefficient for the external influence.
        """
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.clique)].groupby('address')['pct_supply'].sum() / (no_token_communities_snapshot - len(self.clique))
        normalised_pct_supply_externalC = self.sub_dataFrame_sample_population[~self.sub_dataFrame_sample_population.token_address.isin(self.clique)].groupby('address')['pct_supply'].sum() / (no_token_communities_snapshot - len(self.clique))
        normalised_pct_clique_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='gini', alternative='lower')

        self.analysis_result['gini_external_influence'] = gini(normalised_pct_supply_external)
        self.analysis_result_sample_population['gini_external_influence'] = gini(normalised_pct_supply_externalC)
        self.pvalues['gini_external_influence'] = normalised_pct_clique_pval

    def _calculate_total_influence_directional(self):
        """
        Computes the directional total influence of the clique.
        """
        normalised_pct_supply = self.sub_dataFrame.groupby('address')['pct_supply'].sum()
        normalised_pct_supplyC = self.sub_dataFrame_sample_population.groupby('address')['pct_supply'].sum()
        normalised_pct_clique_pval = permutation_test(normalised_pct_supply, normalised_pct_supplyC, method='mean', alternative='greater')

        self.analysis_result['total_influence_directional'] = normalised_pct_supply.sum()
        self.analysis_result_sample_population['total_influence_directional'] = normalised_pct_supplyC.sum()
        self.pvalues['total_influence_directional'] = normalised_pct_clique_pval

    ###########################
    ##### Wealth Metrics ######
    ###########################

    def _calculate_total_wealth(self):
        """
        Computes the total wealth within the clique.
        """
        wealth_clique = self.sub_dataFrame.groupby('address')['value_usd'].sum()
        wealth_cliqueC = self.sub_dataFrame_sample_population.groupby('address')['value_usd'].sum()
        normalised_pct_clique_pval = permutation_test(wealth_clique, wealth_cliqueC, method='mean', alternative='greater')

        self.analysis_result['total_wealth'] = wealth_clique.sum()
        self.analysis_result_sample_population['total_wealth'] = wealth_cliqueC.sum()
        self.pvalues['total_wealth'] = normalised_pct_clique_pval

    def _calculate_gini_of_total_wealth(self):
        """
        Computes the Gini coefficient for the total wealth within the clique.
        """
        gini_wealth_clique = self.sub_dataFrame.groupby('address')['value_usd'].sum()
        gini_wealth_cliqueC = self.sub_dataFrame_sample_population.groupby('address')['value_usd'].sum()
        normalised_pct_clique_pval = permutation_test(gini_wealth_clique, gini_wealth_cliqueC, method='gini', alternative='greater')

        self.analysis_result['gini_total_wealth'] = gini(gini_wealth_clique)
        self.analysis_result_sample_population['gini_total_wealth'] = gini(gini_wealth_cliqueC)
        self.pvalues['gini_total_wealth'] = normalised_pct_clique_pval

    def _calculate_median_wealth(self):
        """
        Computes the median wealth  within the clique.
        """
        median_wealth_clique = self.sub_dataFrame.groupby('address')['value_usd'].sum()
        median_wealth_cliqueC = self.sub_dataFrame_sample_population.groupby('address')['value_usd'].sum()
        median_wealth_clique_pval = permutation_test(median_wealth_clique, median_wealth_cliqueC, method='median', alternative='greater')

        self.analysis_result['median_wealth'] = median_wealth_clique.median()
        self.analysis_result_sample_population['median_wealth'] = median_wealth_cliqueC.median()
        self.pvalues['median_wealth'] = median_wealth_clique_pval

    def _calculate_internal_wealth(self):
        """
        Computes the internal wealth of the clique (value of tokens that are part of the clique).
        """
        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.clique)].groupby('address')['value_usd'].sum()
        normalised_pct_supply_internalC = self.sub_dataFrame_sample_population[self.sub_dataFrame_sample_population.token_address.isin(self.clique)].groupby('address')['value_usd'].sum()
        normalised_pct_clique_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='mean', alternative='greater')

        self.analysis_result['internal_wealth'] = normalised_pct_supply_internal.sum()
        self.analysis_result_sample_population['internal_wealth'] = normalised_pct_supply_internalC.sum()
        self.pvalues['internal_wealth'] = normalised_pct_clique_pval

    def _calculate_gini_of_internal_wealth(self):
        """
        Computes the Gini coefficient for the internal wealth.
        """
        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.clique)].groupby('address')['value_usd'].sum()
        normalised_pct_supply_internalC = self.sub_dataFrame_sample_population[self.sub_dataFrame_sample_population.token_address.isin(self.clique)].groupby('address')['value_usd'].sum()
        normalised_pct_clique_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='gini', alternative='lower')

        self.analysis_result['gini_internal_wealth'] = gini(normalised_pct_supply_internal)
        self.analysis_result_sample_population['gini_internal_wealth'] = gini(normalised_pct_supply_internalC)
        self.pvalues['gini_internal_wealth'] = normalised_pct_clique_pval

    def _calculate_external_wealth(self):
        """
        Computes the external wealth of the clique (wealth in tokens not part of the clique but part of the sample).
        """
        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.clique)].groupby('address')['value_usd'].sum()
        normalised_pct_supply_externalC = self.sub_dataFrame_sample_population[~self.sub_dataFrame_sample_population.token_address.isin(self.clique)].groupby('address')['value_usd'].sum()
        normalised_pct_clique_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='mean', alternative='greater')

        self.analysis_result['external_wealth'] = normalised_pct_supply_external.sum()
        self.analysis_result_sample_population['external_wealth'] = normalised_pct_supply_externalC.sum()
        self.pvalues['external_wealth'] = normalised_pct_clique_pval

    def _calculate_gini_of_external_wealth(self):
        """
        Computes the Gini coefficient for the external wealth.
        """
        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.clique)].groupby('address')['value_usd'].sum()
        normalised_pct_supply_externalC = self.sub_dataFrame_sample_population[~self.sub_dataFrame_sample_population.token_address.isin(self.clique)].groupby('address')['value_usd'].sum()
        normalised_pct_clique_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='gini', alternative='lower')

        self.analysis_result['gini_external_wealth'] = gini(normalised_pct_supply_external)
        self.analysis_result_sample_population['gini_external_wealth'] = gini(normalised_pct_supply_externalC)
        self.pvalues['gini_external_wealth'] = normalised_pct_clique_pval
