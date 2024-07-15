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
    def __init__(self, sub_df, sub_df_control, df, clique, token_lookup):
        self.sub_df = sub_df
        self.sub_df_control = sub_df_control
        self.df = df
        self.clique = clique
        self.token_lookup = token_lookup
        self.analysis_result = {}
        self.analysis_result_control = {}
        self.pvalues = {}

    def analyze(self):
        self._analyze_descriptive_metrics()
        self._analyze_influence_metrics()
        self._analyze_wealth_distribution()
        self._analyze_labels()
        return [self.token_lookup[i] for i in self.clique], self.analysis_result, self.analysis_result_control, self.pvalues

    def _analyze_descriptive_metrics(self):
        self.analysis_result['size_clique'] = self.sub_df.address.nunique()
        self.analysis_result_control['size_clique'] = self.sub_df_control.address.nunique()

    def _analyze_influence_metrics(self):
        self._calculate_total_influence()
        self._calculate_internal_influence()
        self._calculate_external_influence()

    def _calculate_total_influence(self):
        no_token_communities_snapshot = self.df.token_address.nunique()
        normalised_pct_supply = self.sub_df.pct_supply / no_token_communities_snapshot
        normalised_pct_supplyC = self.sub_df_control.pct_supply / no_token_communities_snapshot

        print(f"Sample: {len(normalised_pct_supply)} || Control: {len(normalised_pct_supplyC)}")

        total_influence = normalised_pct_supply.sum()
        total_influence_pval = permutation_test(normalised_pct_supply, normalised_pct_supplyC, method='mean', alternative='greater')

        self.analysis_result['total_influence'] = total_influence
        self.analysis_result_control['total_influence'] = normalised_pct_supplyC.sum()
        self.pvalues['total_influence'] = total_influence_pval

        gini_total_influence = gini(normalised_pct_supply)
        gini_total_influenceC = gini(normalised_pct_supplyC)
        gini_total_influence_pval = permutation_test(normalised_pct_supply, normalised_pct_supplyC, method='gini', alternative="lower")

        self.analysis_result['gini_total_influence'] = gini_total_influence
        self.analysis_result_control['gini_total_influence'] = gini_total_influenceC
        self.pvalues['gini_total_influence'] = gini_total_influence_pval

    def _calculate_internal_influence(self):
        no_token_communities_in_clique = len(self.clique)
        normalised_pct_supply_internal = self.sub_df[self.sub_df.token_address.isin(self.clique)].pct_supply / no_token_communities_in_clique
        normalised_pct_supply_internalC = self.sub_df_control[self.sub_df_control.token_address.isin(self.clique)].pct_supply / no_token_communities_in_clique

        internal_influence = normalised_pct_supply_internal.sum()
        internal_influence_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='mean', alternative='greater')

        self.analysis_result['internal_influence'] = internal_influence
        self.analysis_result_control['internal_influence'] = normalised_pct_supply_internalC.sum()
        self.pvalues['internal_influence'] = internal_influence_pval

        gini_internal_influence = gini(normalised_pct_supply_internal)
        gini_internal_influenceC = gini(normalised_pct_supply_internalC)
        gini_internal_influence_pval = permutation_test(normalised_pct_supply_internal, normalised_pct_supply_internalC, method='gini', alternative="lower")

        self.analysis_result['gini_internal_influence'] = gini_internal_influence
        self.analysis_result_control['gini_internal_influence'] = gini_internal_influenceC
        self.pvalues['gini_internal_influence'] = gini_internal_influence_pval

    def _calculate_external_influence(self):
        no_token_communities_snapshot = self.df.token_address.nunique()
        no_token_communities_in_clique = len(self.clique)
        no_token_communities_not_in_clique = no_token_communities_snapshot - no_token_communities_in_clique

        normalised_pct_supply_external = self.sub_df[~self.sub_df.token_address.isin(self.clique)].pct_supply / no_token_communities_not_in_clique
        normalised_pct_supply_externalC = self.sub_df_control[~self.sub_df_control.token_address.isin(self.clique)].pct_supply / no_token_communities_not_in_clique

        external_influence = normalised_pct_supply_external.sum()
        external_influence_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='mean', alternative='greater')

        self.analysis_result['external_influence'] = external_influence
        # self.analysis_result_control['external_influence'] = normalised_pct_supply_externalC.sum()
        self.pvalues['external_influence'] = external_influence_pval

        gini_external_influence = gini(normalised_pct_supply_external)
        # gini_external_influenceC = gini(normalised_pct_supply_externalC)
        gini_external_influence_pval = permutation_test(normalised_pct_supply_external, normalised_pct_supply_externalC, method='gini', alternative="lower")

        self.analysis_result['gini_external_influence'] = gini_external_influence
        # self.analysis_result_control['gini_external_influence'] = gini_external_influenceC
        self.pvalues['gini_external_influence'] = gini_external_influence_pval

    def _analyze_wealth_distribution(self):
        self._calculate_wealth_clique()
        self._calculate_gini_wealth_clique()
        self._calculate_median_wealth_level()
        self._calculate_median_no_assets()

    def _calculate_wealth_clique(self):
        wealth_clique = self.sub_df['value_usd'].sum()
        total_wealth_pval = permutation_test(self.sub_df['value_usd'], self.sub_df_control['value_usd'], method='median', alternative='greater')

        self.analysis_result['wealth_clique'] = wealth_clique
        self.analysis_result_control['wealth_clique'] = self.sub_df_control['value_usd'].sum()
        self.pvalues['wealth_clique'] = total_wealth_pval

    def _calculate_gini_wealth_clique(self):
        gini_wealth_clique = gini(self.sub_df['value_usd'])
        gini_wealth_cliqueC = gini(self.sub_df_control['value_usd'])
        gini_wealth_clique_pval =permutation_test(self.sub_df['value_usd'], self.sub_df_control['value_usd'], method='gini', alternative="lower")

        self.analysis_result['gini_wealth_clique'] = gini_wealth_clique
        self.analysis_result_control['gini_wealth_clique'] = gini_wealth_cliqueC
        self.pvalues['gini_wealth_clique'] = gini_wealth_clique_pval

    def _calculate_median_wealth_level(self):
        median_wealth_level_clique = self.sub_df.groupby('address')['value_usd'].sum().median()
        median_wealth_level_cliqueC = self.sub_df_control.groupby('address')['value_usd'].sum().median()
        median_wealth_level_clique_pval = permutation_test(self.sub_df.groupby('address')['value_usd'].sum(), self.sub_df_control.groupby('address')['value_usd'].sum(), method='median', alternative='greater')

        self.analysis_result['median_wealth_level_clique'] = median_wealth_level_clique
        self.analysis_result_control['median_wealth_level_clique'] = median_wealth_level_cliqueC
        self.pvalues['median_wealth_level_clique'] = median_wealth_level_clique_pval

    def _calculate_median_no_assets(self):
        median_no_assets_clique = self.sub_df.groupby('address')['token_address'].count().median()
        median_no_assets_cliqueC = self.sub_df_control.groupby('address')['token_address'].count().median()
        median_no_assets_clique_pval = permutation_test(self.sub_df.groupby('address')['token_address'].count(), self.sub_df_control.groupby('address')['token_address'].count(),method='median', alternative='greater')

        self.analysis_result['median_no_assets_clique'] = median_no_assets_clique
        self.analysis_result_control['median_no_assets_clique'] = median_no_assets_cliqueC
        self.pvalues['median_no_assets_clique'] = median_no_assets_clique_pval

    def _analyze_labels(self):
        # self.sub_df.label.fillna('other_contracts', inplace=True)
        # self.sub_df_control.label.fillna('other_contracts', inplace=True)

        self.sub_df.fillna({'label': 'other_contracts'}, inplace=True)
        self.sub_df_control.fillna({'label': 'other_contracts'}, inplace=True)
        self.analysis_result['max_influence_label_distribution'] = dict(self.sub_df.groupby(['label'])['pct_supply'].sum() / self.df.token_address.nunique())
        self.analysis_result_control['max_influence_label_distribution'] = dict(self.sub_df_control.groupby(['label'])['pct_supply'].sum() / self.df.token_address.nunique())

def clique_member_wallets_weak(clique, dataFrame):
    clique_members = set()
    for c in combinations(clique, 2):
        set1_addresses = set(dataFrame[dataFrame.token_address == c[0]].address.unique())
        set2_addresses = set(dataFrame[dataFrame.token_address == c[1]].address.unique())
        clique_members.update(set1_addresses & set2_addresses)
    return list(clique_members)

def clique_member_wallets_strong(clique, dataFrame):
    full_member_addresses = set(dataFrame[dataFrame.token_address == clique[0]].address.unique())
    for token in clique[1:]:
        token_addresses = set(dataFrame[dataFrame.token_address == token].address.unique())
        full_member_addresses &= token_addresses
        if not full_member_addresses:
            return []
    return list(full_member_addresses)

def analyze_clique(sub_dataFrame, sub_dataFrame_control, dataFrame, clique, token_lookup):
    analyzer = CliqueAnalysis(sub_dataFrame, sub_dataFrame_control, dataFrame, clique, token_lookup)
    return analyzer.analyze()
