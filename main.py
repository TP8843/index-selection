import pandas as pd
import numpy as np
from pandas import DataFrame
from itertools import combinations
import rustworkx as rx

# Defines beta for similarity cutoff
B = 0.5

def main():
    classes = pd.read_csv("./class.csv")["Class_Type"].tolist()
    df: DataFrame = pd.read_csv("./zoo.csv")

    num_features: int  = df.columns
    l: int = len(num_features)
    cutoff: int = l * (1-B)

    print("Cutoff for graph similarity:", cutoff)

    graph: rx.PyGraph = rx.PyGraph()

    for index in range(df.shape[0]):
        graph.add_node(index)

    print("Finished adding vertices to graph")

    c : list[int] = list(combinations(graph.node_indexes(), 2))

    for indexes in c:
        i1: np.ndarray[num_features] = df.iloc[indexes[0]].to_numpy()
        i2: np.ndarray[num_features] = df.iloc[indexes[1]].to_numpy()

        distance: int = np.count_nonzero(i1 != i2)

        # Add dissimilar edges to graph
        if distance > cutoff:
            graph.add_edge(indexes[0], indexes[1], distance)

    print("Finished adding edges to graph")

    indexes: list[int] = []
    working_graph = graph.copy()

    while len(working_graph.node_indexes()) > 0:
        degrees: list[(int, int)] = [(index, working_graph.degree(index) ) for index in working_graph.node_indexes()]
        degrees: list[(int, int)] = sorted(degrees, key=lambda x: x[1])

        neighbours: list[int] = working_graph.neighbors(degrees[0][0])
        indexes.append(degrees[0][0])

        working_graph.remove_node(degrees[0][0])
        working_graph.remove_nodes_from(neighbours)

    print("Final stable set size:", str(len(indexes)), "/", str(df.shape[0]))

    filtered_df = df.iloc[indexes]
    print(filtered_df)
    filtered_df.to_csv("./filtered_zoo.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
