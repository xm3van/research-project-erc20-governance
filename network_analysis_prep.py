# imports

import pandas as pd
import numpy as np

import dask.dataframe as dd
import dask.array as da
import dask.bag as db
import itertools

from os.path import join
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests as m_tests
import matplotlib.pyplot as plt
from ast import literal_eval
import networkx as nx
import os

from dotenv import load_dotenv

load_dotenv()

path = os.environ["PROJECT_PATH"]


summary_stats = {}


progress = 0

# snapshot selection
df_snapshot = pd.read_csv("assets/snapshot_selection.csv")

# address selection
df_addresses = pd.read_csv("assets/df_final_token_selection_20221209.csv")

# burner addresses
# remove burner addresses
known_burner_addresses = [
    "0x0000000000000000000000000000000000000000",
    "0x0000000000000000000000000000000000000000",
    "0x0000000000000000000000000000000000000001",
    "0x0000000000000000000000000000000000000002",
    "0x0000000000000000000000000000000000000003",
    "0x0000000000000000000000000000000000000004",
    "0x0000000000000000000000000000000000000005",
    "0x0000000000000000000000000000000000000006",
    "0x0000000000000000000000000000000000000007",
    "0x000000000000000000000000000000000000dead",
]


#### functions ####


def main(address1, address2, pop_size):

    # get unique addresses
    token1_uniqa = ddf[ddf.token_address == address1].address.unique()
    token2_uniqa = ddf[ddf.token_address == address2].address.unique()

    # calcluate intersection
    token1_token2_uniqa_intersection = np.intersect1d(
        token1_uniqa, token2_uniqa, assume_unique=True
    )

    # calcualte number
    len_token1 = len(token1_uniqa)
    len_token2 = len(token2_uniqa)
    len_intersection = len(token1_token2_uniqa_intersection)

    # calculate hyptoge

    # Define the parameters of the distribution
    M = pop_size  # population size
    n = len_token1  # number of draws
    K = len_token2  # number of successes in population
    x = len_intersection  # number of successes in draws

    # Compute the cumulative probability of obtaining at most x successes
    pvalue = 1 - hypergeom.cdf(x, M, n, K)

    # print(f'token_address {address1} has {len_token1} Unique Addresses | token_address {address2} has {len_token2} Unique Addresses | Intersection: {len_intersection} | p value: {pvalue}')

    return pvalue


###################


for snapshot in df_snapshot["Block Height"]:

    # Info
    items_left = len(df_snapshot) - progress
    progress += 1
    print(
        f"Current Snapshot: {snapshot} || Items processed: {progress} || Items left: { (items_left) }"
    )

    ## formating of data
    # load data
    ddf = dd.read_csv(
        join(
            path,
            f"token_balance_lookup_tables/token_holder_snapshot_balance_{snapshot}.csv",
        )
    )

    # filter data
    ddf = ddf[ddf.value > 0]
    ddf = ddf[ddf.token_address.isin(df_addresses.address) == True]

    # remove known burner addresses
    ddf = ddf[ddf.address.isin(known_burner_addresses) == False]

    # population size
    pop_size = len(ddf.address.unique())
    p_dict = {}

    # reduce address list

    present_addresses = list(ddf.token_address.unique().compute())

    if len(present_addresses) == 1:

        print("One address only")
        stats = {
            "nodes": np.nan,
            "possible_nodes": len(present_addresses),
            "edges": np.nan,
            "avg_degree_path": np.nan,
            "min_degree_path": np.nan,
            "max_degree_path": np.nan,
            "diameter": np.nan,
            "avg_shortest_path": np.nan,
            "density": np.nan,
            "degree_centrality_avg": np.nan,
            "degree_centrality_min": np.nan,
            "degree_centrality_max": np.nan,
            "betweeness_centrality_avg": np.nan,
            "betweeness_centrality_min": np.nan,
            "betweeness_centrality_max": np.nan,
            "triangles_avg": np.nan,
            "triangles_min": np.nan,
            "triangles_max": np.nan,
        }

    else:

        try:
            # iterations
            for combination in itertools.combinations(present_addresses, 2):

                pvalue = main(combination[0], combination[1], pop_size)

                p_dict[combination] = pvalue

            ## Evaluate pvalues
            # store pvalues
            df_pvalues = pd.DataFrame.from_dict(p_dict, orient="index")
            df_pvalues.reset_index(inplace=True)
            df_pvalues.columns = ["combination", "p_value"]

            # value test
            m_test = m_tests(pvals=df_pvalues.p_value, alpha=0.01, method="bonferroni")
            df_pvalues["m_test_result"] = m_test[0]
            df_pvalues["m_test_value"] = m_test[1]
            df_pvalues.to_csv(join(path, f"output/pvalues_{snapshot}.csv"))

            ## Build graph
            # filter df
            df_pvalues_validated = df_pvalues[df_pvalues.m_test_result == True]

            # Create an empty graph
            G = nx.Graph()

            # Add the edges to the graph
            G.add_edges_from(df_pvalues_validated.combination)

            # create labels
            df_a_fil = df_addresses[df_addresses.address.isin(list(G.nodes()))]
            labels = (
                df_a_fil[["address", "name"]].set_index("address").to_dict()["name"]
            )

            # visualise netwotk
            nx.draw(G, labels=labels)

            # show
            plt.savefig(join(path, f"output/pics/pic_vNetwork_{snapshot}.png"))

            # clear img
            plt.clf()

            ## descriptve statistic
            g_nodes = G.number_of_nodes()
            g_edges = G.number_of_edges()
            ## degree
            g_degrees = dict(G.degree())
            # Calculate the average degree of the graph

            try:
                g_avg_degree = sum(g_degrees.values()) / len(g_degrees)
                g_max_degree = max(g_degrees.values())
                g_min_degree = min(g_degrees.values())
            except:
                g_avg_degree = np.nan
                g_max_degree = np.nan
                g_min_degree = np.nan

            # diameter
            try:
                g_diameter = nx.diameter(G)
            except:
                g_diameter = np.nan

            # average shortest path
            try:
                g_avg_shortest_path = nx.average_shortest_path_length(G)
            except:
                g_avg_shortest_path = np.nan

            g_density = nx.density(G)

            # centrality
            g_degree_centrality = dict(nx.degree_centrality(G))

            try:
                g_degree_centrality_avg = sum(g_degree_centrality.values()) / len(
                    g_degree_centrality
                )
                g_degree_centrality_min = min(g_degree_centrality.values())
                g_degree_centrality_max = max(g_degree_centrality.values())

            except:

                g_degree_centrality_avg = np.nan
                g_degree_centrality_min = np.nan
                g_degree_centrality_max = np.nan

            # betweeness
            g_betweeness_centrality = dict(nx.betweenness_centrality(G))

            try:
                g_betweeness_centrality_avg = sum(
                    g_betweeness_centrality.values()
                ) / len(g_betweeness_centrality)
                g_betweeness_centrality_min = min(g_betweeness_centrality.values())
                g_betweeness_centrality_max = max(g_betweeness_centrality.values())

            except:
                g_betweeness_centrality_avg = np.nan
                g_betweeness_centrality_min = np.nan
                g_betweeness_centrality_max = np.nan

            # triangles
            g_triangles = dict(nx.triangles(G))

            try:
                g_triangles_avg = sum(g_triangles.values()) / len(g_triangles)
                g_triangles_min = min(g_triangles.values())
                g_triangles_max = max(g_triangles.values())
            except:
                g_triangles_avg = np.nan
                g_triangles_min = np.nan
                g_triangles_max = np.nan

            # g_clustering = dict(nx.clustering(G))
            # g_clustering_avg = sum(g_clustering.values())/ len(g_clustering)
            # g_clustering_min = min(g_clustering.values())
            # g_clustering_max = max(g_clustering.values())

            stats = {
                "nodes": g_nodes,
                "possible_nodes": len(present_addresses),
                "edges": g_edges,
                "avg_degree_path": g_avg_degree,
                "min_degree_path": g_min_degree,
                "max_degree_path": g_max_degree,
                "diameter": g_diameter,
                "avg_shortest_path": g_avg_shortest_path,
                "density": g_density,
                "degree_centrality_avg": g_degree_centrality_avg,
                "degree_centrality_min": g_degree_centrality_min,
                "degree_centrality_max": g_degree_centrality_max,
                "betweeness_centrality_avg": g_betweeness_centrality_avg,
                "betweeness_centrality_min": g_betweeness_centrality_min,
                "betweeness_centrality_max": g_betweeness_centrality_max,
                "triangles_avg": g_triangles_avg,
                "triangles_min": g_triangles_min,
                "triangles_max": g_triangles_max,
            }
            # 'clustering_avg': g_clustering_avg, 'clustering_min': g_clustering_min, 'clustering_max': g_clustering_max }

        except:
            stats = {
                "nodes": np.nan,
                "possible_nodes": len(present_addresses),
                "edges": np.nan,
                "avg_degree_path": np.nan,
                "min_degree_path": np.nan,
                "max_degree_path": np.nan,
                "diameter": np.nan,
                "avg_shortest_path": np.nan,
                "density": np.nan,
                "degree_centrality_avg": np.nan,
                "degree_centrality_min": np.nan,
                "degree_centrality_max": np.nan,
                "betweeness_centrality_avg": np.nan,
                "betweeness_centrality_min": np.nan,
                "betweeness_centrality_max": np.nan,
                "triangles_avg": np.nan,
                "triangles_min": np.nan,
                "triangles_max": np.nan,
            }
        # Reseit Graph
        G = nx.Graph()

    summary_stats[snapshot] = stats


df_stats = pd.DataFrame.from_dict(summary_stats)
df_stats.to_csv("stats.csv")
