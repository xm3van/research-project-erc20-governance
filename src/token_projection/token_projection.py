import networkx as nx

def compute_weighted_degree(G, alpha=0.5, weight='weight'):
    """
    Compute the weighted degree of each node in a graph based on the given formula:
    kw_i = alpha * k_i + (1 - alpha) * sum(w_ij), for all j in neighbors(i)

    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    alpha : float, optional
        The tuning parameter used in the weighted degree formula. Default is 0.5.
    weight : str, optional
        The name of the edge attribute used as weight. Default is 'weight'.

    Returns:
    --------
    weighted_degrees : dict
        A dictionary where keys are nodes and values are the computed weighted degrees.
        
    Implementation Reference: 
    -------------------------
    Wei, B., Liu, J., Wei, D., Gao, C., & Deng, Y. (2015). Weighted k-shell decomposition for
    complex networks based on potential edge weights. Physica A: Statistical Mechanics and its
    Applications, 420, 277-283.
    """
    # Dictionary to hold the weighted degree of each node
    weighted_degrees = {}
    
    for node in G.nodes():
        # Get the degree of the node
        k_i = G.degree(node)
        
        # Get the sum of the weights of the edges connected to the node
        sum_weights = sum(data.get(weight, 1) for _, _, data in G.edges(node, data=True))
        
        # Compute the weighted degree of the node using the formula
        k_w_i = alpha * k_i + (1 - alpha) * sum_weights
        
        # Store the weighted degree in the dictionary
        weighted_degrees[node] = k_w_i
    
    return weighted_degrees

def weighted_k_core(G, k, alpha=0.5, weight='weight'):
    """
    Compute the weighted k-core of a graph based on the provided weighted degree formula.

    Parameters:
    -----------
    G : networkx.Graph
        The input graph.
    k : float
        The minimum weighted degree for nodes to be kept.
    alpha : float, optional
        The tuning parameter used in the weighted degree formula. Default is 0.5.
    weight : str, optional
        The name of the edge attribute used as weight. Default is 'weight'.

    Returns:
    --------
    core_subgraph : networkx.Graph
        The weighted k-core subgraph.
    """
    # Create a copy of the graph to avoid modifying the original graph
    core_subgraph = G.copy()
    
    while True:
        # Compute the weighted degree of each node
        weighted_degrees = compute_weighted_degree(core_subgraph, alpha, weight)
        
        # Identify nodes to be removed based on the weighted degree
        to_remove = [node for node, deg in weighted_degrees.items() if deg < k]
        
        # If no nodes to remove, the current core is the final core
        if not to_remove:
            break
        
        # Remove nodes from the graph
        core_subgraph.remove_nodes_from(to_remove)
    
    return core_subgraph