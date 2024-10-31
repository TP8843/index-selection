from collections.abc import Callable

import numpy as np
import pandas as pd
import rustworkx as rx

def generate_graph(df: pd.DataFrame, beta: float) -> rx.PyGraph:
    # Generate variables
    l: int = len(df.columns)
    num_indices = df.shape[0]

    # At what distance value to draw an edge
    min_distance: float = l * (1-beta)
    print("Minimum distance for edges:", min_distance)

    graph: rx.PyGraph = rx.PyGraph()
    # Adds nodes representing all rows of data
    graph.add_nodes_from(range(num_indices))

    for i in range(num_indices):
        for j in range(i, num_indices):
            distance: int = np.count_nonzero(df.iloc[i] != df.iloc[j])
            if distance >= min_distance:
                graph.add_edge(i, j, distance)

    return graph

def get_stable_set(df: pd.DataFrame, beta: float, selection_function: Callable[[pd.DataFrame, rx.PyGraph], int]) -> list[int]:
    working_graph = generate_graph(df, beta)
    indexes: list[int] = []

    while len(working_graph.node_indexes()) > 0:
        node: int = selection_function(df, working_graph)
        neighbours: list[int] = working_graph.neighbors(node)

        working_graph.remove_node(node)
        working_graph.remove_nodes_from(neighbours)

        indexes.append(node)

    return indexes

def minimum_degree_selection(df: pd.DataFrame, graph: rx.PyGraph) -> int:
    # Bigger than the largest possible degree
    min_degree: int = df.shape[0]
    index: int

    for i in graph.node_indexes():
        degree = graph.degree(i)
        if degree < min_degree:
            min_degree = degree
            index = i

    return index
