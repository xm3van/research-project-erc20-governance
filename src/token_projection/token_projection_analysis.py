# imports
import pandas as pd
import numpy as np
from scipy.stats import hypergeom 
from statsmodels.stats.multitest import multipletests
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from os.path import join

def calculate_pvalue(address1, address2, ddf, pop_size):
    token1_uniqa = ddf[ddf.token_address == address1].address.unique()
    token2_uniqa = ddf[ddf.token_address == address2].address.unique()
    intersection = np.intersect1d(token1_uniqa, token2_uniqa, assume_unique=True)
    pvalue = 1 - hypergeom.cdf(len(intersection), pop_size, len(token1_uniqa), len(token2_uniqa))
    return pvalue


def validate_token_links(present_addresses, ddf, pop_size):
    if len(present_addresses) <= 1:
        return {'nodes': np.nan, 'possible_nodes': len(present_addresses)}

    p_dict = {(a1, a2): calculate_pvalue(a1, a2, ddf, pop_size) for a1, a2 in combinations(present_addresses, 2)}
    df_pvalues = pd.DataFrame(list(p_dict.items()), columns=['combination', 'p_value'])
    # Adjust p-values
    df_pvalues['m_test_result'], df_pvalues['m_test_value'] = multipletests(df_pvalues.p_value, alpha=0.01, method='bonferroni')[:2]
    return df_pvalues[df_pvalues.m_test_result]







# def calculate_network_metrics(G):
#     # Calculate basic network metrics
#     metrics = {
#         'nodes': G.number_of_nodes(),
#         'edges': G.number_of_edges(),
#         'density': nx.density(G),
#         'diameter': nx.diameter(G) if nx.is_connected(G) else 'Graph is not connected',
#         'avg_shortest_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else 'Graph is not connected'
#     }

#     # Degree Centrality
#     degree_centrality = nx.degree_centrality(G)
#     metrics['degree_centrality_avg'] = sum(degree_centrality.values()) / len(degree_centrality)

#     # Betweenness Centrality
#     betweenness_centrality = nx.betweenness_centrality(G)
#     metrics['betweenness_centrality_avg'] = sum(betweenness_centrality.values()) / len(betweenness_centrality)

#     # Clustering Coefficient
#     clustering_coefficient = nx.clustering(G)
#     metrics['clustering_coefficient_avg'] = sum(clustering_coefficient.values()) / len(clustering_coefficient)

#     # Triangles
#     triangles = nx.triangles(G)
#     metrics['triangles_avg'] = sum(triangles.values()) / len(triangles)

#     return metrics

# def analyze_graph(df_pvalues_validated):
#     G = nx.Graph()
#     G.add_edges_from(df_pvalues_validated.combination)

#     # Ensure the graph is not empty
#     if len(G) == 0:
#         return {
#             'nodes': 0,
#             'edges': 0,
#             'density': 'N/A',
#             'diameter': 'N/A',
#             'avg_shortest_path_length': 'N/A',
#             'degree_centrality_avg': 'N/A',
#             'betweenness_centrality_avg': 'N/A',
#             'clustering_coefficient_avg': 'N/A',
#             'triangles_avg': 'N/A'
#         }
    
#     # If the graph is not connected, consider using connected components for some metrics
#     if not nx.is_connected(G):
#         largest_cc = max(nx.connected_components(G), key=len)
#         G = G.subgraph(largest_cc).copy()

#     return calculate_network_metrics(G)



# def compute_weighted_degree(G, alpha=0.5, weight='weight'):
#     """
#     Compute the weighted degree of each node in a graph based on the given formula:
#     kw_i = alpha * k_i + (1 - alpha) * sum(w_ij), for all j in neighbors(i)

#     Parameters:
#     -----------
#     G : networkx.Graph
#         The input graph.
#     alpha : float, optional
#         The tuning parameter used in the weighted degree formula. Default is 0.5.
#     weight : str, optional
#         The name of the edge attribute used as weight. Default is 'weight'.

#     Returns:
#     --------
#     weighted_degrees : dict
#         A dictionary where keys are nodes and values are the computed weighted degrees.
        
#     Implementation Reference: 
#     -------------------------
#     Wei, B., Liu, J., Wei, D., Gao, C., & Deng, Y. (2015). Weighted k-shell decomposition for
#     complex networks based on potential edge weights. Physica A: Statistical Mechanics and its
#     Applications, 420, 277-283.
#     """
#     # Dictionary to hold the weighted degree of each node
#     weighted_degrees = {}
    
#     for node in G.nodes():
#         # Get the degree of the node
#         k_i = G.degree(node)
        
#         # Get the sum of the weights of the edges connected to the node
#         sum_weights = sum(data.get(weight, 1) for _, _, data in G.edges(node, data=True))
        
#         # Compute the weighted degree of the node using the formula
#         k_w_i = alpha * k_i + (1 - alpha) * sum_weights
        
#         # Store the weighted degree in the dictionary
#         weighted_degrees[node] = k_w_i
    
#     return weighted_degrees

# def weighted_k_core(G, k, alpha=0.5, weight='weight'):
#     """
#     Compute the weighted k-core of a graph based on the provided weighted degree formula.

#     Parameters:
#     -----------
#     G : networkx.Graph
#         The input graph.
#     k : float
#         The minimum weighted degree for nodes to be kept.
#     alpha : float, optional
#         The tuning parameter used in the weighted degree formula. Default is 0.5.
#     weight : str, optional
#         The name of the edge attribute used as weight. Default is 'weight'.

#     Returns:
#     --------
#     core_subgraph : networkx.Graph
#         The weighted k-core subgraph.
#     """
#     # Create a copy of the graph to avoid modifying the original graph
#     core_subgraph = G.copy()
    
#     while True:
#         # Compute the weighted degree of each node
#         weighted_degrees = compute_weighted_degree(core_subgraph, alpha, weight)
        
#         # Identify nodes to be removed based on the weighted degree
#         to_remove = [node for node, deg in weighted_degrees.items() if deg < k]
        
#         # If no nodes to remove, the current core is the final core
#         if not to_remove:
#             break
        
#         # Remove nodes from the graph
#         core_subgraph.remove_nodes_from(to_remove)
    
#     return core_subgraph