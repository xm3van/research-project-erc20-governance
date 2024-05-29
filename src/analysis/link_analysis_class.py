from scipy.stats import hypergeom 
import pandas as pd
import numpy as np

from src.utilities.metrics_and_tests import *
from src.utilities.utils import * 

from itertools import combinations

class LinkAnalysis:
    def __init__(self, link, dataFrame, sub_dataFrame, sub_dataFrame_control, token_lookup):
        self.link = link
        self.dataFrame = dataFrame
        self.sub_dataFrame = sub_dataFrame
        self.sub_dataFrame_control = sub_dataFrame_control
        self.token_lookup = token_lookup
        self.analysis_result = {}
        self.analysis_result_control = {}
        self.pvalues = {}

    def link_member_wallets(self):
        """
        Retrieve unique wallet addresses that hold ALL tokens in the provided link.
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
        """
        self._analyze_descriptive_metrics()
        self._analyze_influence_metrics()
        self._analyze_wealth_distribution()
        self._analyze_labels()
        return [self.token_lookup[i] for i in self.link], self.analysis_result, self.analysis_result_control, self.pvalues

    def _analyze_descriptive_metrics(self):
        self.analysis_result['size_link'] = self.sub_dataFrame.address.nunique()
        self.analysis_result_control['size_link'] = self.sub_dataFrame_control.address.nunique()

    def _analyze_influence_metrics(self):
        no_token_communities_snapshot = self.dataFrame.token_address.nunique()

        normalised_pct_supply = self.sub_dataFrame.pct_supply / no_token_communities_snapshot
        normalised_pct_supplyC = self.sub_dataFrame_control.pct_supply / no_token_communities_snapshot

        self._calculate_influence('total_influence', normalised_pct_supply, normalised_pct_supplyC)
        self._calculate_gini('gini_total_influence', normalised_pct_supply, normalised_pct_supplyC)

        normalised_pct_supply_internal = self.sub_dataFrame[self.sub_dataFrame.token_address.isin(self.link)].pct_supply / len(self.link)
        normalised_pct_supply_internalC = self.sub_dataFrame_control[self.sub_dataFrame_control.token_address.isin(self.link)].pct_supply / len(self.link)

        self._calculate_influence('internal_influence', normalised_pct_supply_internal, normalised_pct_supply_internalC)
        self._calculate_gini('gini_internal_influence', normalised_pct_supply_internal, normalised_pct_supply_internalC)

        normalised_pct_supply_external = self.sub_dataFrame[~self.sub_dataFrame.token_address.isin(self.link)].pct_supply / (no_token_communities_snapshot - len(self.link))
        normalised_pct_supply_externalC = self.sub_dataFrame_control[~self.sub_dataFrame_control.token_address.isin(self.link)].pct_supply / (no_token_communities_snapshot - len(self.link))

        self._calculate_influence('external_influence', normalised_pct_supply_external, normalised_pct_supply_externalC)
        self._calculate_gini('gini_external_influence', normalised_pct_supply_external, normalised_pct_supply_externalC)

    def _calculate_influence(self, metric_name, supply, supply_control):
        influence = supply.sum()
        influence_pval = t_test(supply, supply_control, alternative='greater')

        self.analysis_result[metric_name] = influence
        self.analysis_result_control[metric_name] = supply_control.sum()
        self.pvalues[metric_name] = influence_pval

    def _calculate_gini(self, metric_name, supply, supply_control):
        gini_value = gini(supply)
        gini_value_control = gini(supply_control)
        gini_pval = one_tailed_gini_t_test(supply, supply_control, alpha=0.05, direction="lower")

        self.analysis_result[metric_name] = gini_value
        self.analysis_result_control[metric_name] = gini_value_control
        self.pvalues[metric_name] = gini_pval

    def _analyze_wealth_distribution(self):
        wealth_link = self.sub_dataFrame['value_usd'].sum()
        total_wealth_pval = t_test(self.sub_dataFrame['value_usd'], self.sub_dataFrame_control['value_usd'], alternative='greater')

        self.analysis_result['wealth_link'] = wealth_link
        self.analysis_result_control['wealth_link'] = self.sub_dataFrame_control['value_usd'].sum()
        self.pvalues['wealth_link'] = total_wealth_pval

        gini_wealth_link = gini(self.sub_dataFrame['value_usd'])
        gini_wealth_linkC = gini(self.sub_dataFrame_control['value_usd'])
        gini_wealth_link_pval = one_tailed_gini_t_test(self.sub_dataFrame['value_usd'], self.sub_dataFrame_control['value_usd'], alpha=0.05, direction="lower")

        self.analysis_result['gini_wealth_link'] = gini_wealth_link
        self.analysis_result_control['gini_wealth_link'] = gini_wealth_linkC
        self.pvalues['gini_wealth_link'] = gini_wealth_link_pval

        median_wealth_level_link = self.sub_dataFrame.groupby('address')['value_usd'].sum().median()
        median_wealth_level_linkC = self.sub_dataFrame_control.groupby('address')['value_usd'].sum().median()
        median_wealth_level_link_pval = median_t_test(self.sub_dataFrame.groupby('address')['value_usd'].sum(), self.sub_dataFrame_control.groupby('address')['value_usd'].sum(), alternative='greater')

        self.analysis_result['median_wealth_level_link'] = median_wealth_level_link
        self.analysis_result_control['median_wealth_level_link'] = median_wealth_level_linkC
        self.pvalues['median_wealth_level_link'] = median_wealth_level_link_pval

        median_no_assets_link = self.sub_dataFrame.groupby('address')['token_address'].count().median()
        median_no_assets_linkC = self.sub_dataFrame_control.groupby('address')['token_address'].count().median()
        median_no_assets_link_pval = median_t_test(self.sub_dataFrame.groupby('address')['token_address'].count(), self.sub_dataFrame_control.groupby('address')['token_address'].count(), alternative='greater')

        self.analysis_result['median_no_assets_link'] = median_no_assets_link
        self.analysis_result_control['median_no_assets_link'] = median_no_assets_linkC
        self.pvalues['median_no_assets_link'] = median_no_assets_link_pval

    def _analyze_labels(self):
        self.sub_dataFrame.label.fillna('other_contracts', inplace=True)
        self.sub_dataFrame_control.label.fillna('other_contracts', inplace=True)

        self._calculate_max_influence_label()

    def _calculate_max_influence_label(self):
        max_influence_label = self.sub_dataFrame.groupby(['label'])['pct_supply'].sum().idxmax()
        max_influence_labelC = self.sub_dataFrame_control.groupby(['label'])['pct_supply'].sum().idxmax()

        max_influence_label_value = self.sub_dataFrame.groupby(['label'])['pct_supply'].sum().max() / self.dataFrame.token_address.nunique()
        max_influence_label_valueC = self.sub_dataFrame_control.groupby(['label'])['pct_supply'].sum().max() / self.dataFrame.token_address.nunique()

        sample = self.sub_dataFrame[self.sub_dataFrame.label == max_influence_label].pct_supply
        control = self.sub_dataFrame_control[self.sub_dataFrame_control.label == max_influence_label].pct_supply
        max_influence_label_pval = t_test(sample, control, alternative='greater')

        self.analysis_result['max_influence_label'] = (str(max_influence_label), max_influence_label_value)
        self.analysis_result_control['max_influence_label'] = (str(max_influence_labelC), max_influence_label_valueC)
        self.pvalues['max_influence_label'] = max_influence_label_pval

        self.analysis_result['max_influence_label_distribution'] = dict(self.sub_dataFrame.groupby(['label'])['pct_supply'].sum() / self.dataFrame.token_address.nunique())
        self.analysis_result_control['max_influence_label_distribution'] = dict(self.sub_dataFrame_control.groupby(['label'])['pct_supply'].sum() / self.dataFrame.token_address.nunique())

def analyze_link(sub_dataFrame, sub_dataFrame_control, dataFrame, link, token_lookup):
    analyzer = LinkAnalysis(link, dataFrame, sub_dataFrame, sub_dataFrame_control, token_lookup)
    return analyzer.analyze_link()
