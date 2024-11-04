import time

import pandas as pd
from pandas import DataFrame

from graph import get_stable_set, minimum_degree_selection, get_batches

# Defines beta for similarity cutoff
B = 0.7

def stable_set():
    start_time = time.time()
    classes = pd.read_csv("./class.csv")["Class_Type"].tolist()
    df: DataFrame = pd.read_csv("./zoo.csv")

    indexes: list[int] = get_stable_set(df, B, minimum_degree_selection)

    print("Final stable set size:", str(len(indexes)), "/", str(df.shape[0]))
    print("Time taken to find set:", time.time() - start_time)

    filtered_df = df.iloc[indexes]
    print(filtered_df)
    filtered_df.to_csv("./filtered_zoo.csv")


def batches():
    start_time = time.time()
    classes = pd.read_csv("./class.csv")["Class_Type"].tolist()
    df: DataFrame = pd.read_csv("./zoo.csv")

    batches_list: list[int, int] = get_batches(df, B)

    print("Time taken to find batches:", time.time() - start_time)

    for i in range(len(batches_list)):
        print("Batch", i)
        print(df.iloc[batches_list[i]])
        print("")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    batches()
