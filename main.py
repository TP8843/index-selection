import time

import pandas as pd
from pandas import DataFrame

from graph import get_stable_set, minimum_degree_selection

# Defines beta for similarity cutoff
B = 0.4

def main():
    startTime = time.time()
    classes = pd.read_csv("./class.csv")["Class_Type"].tolist()
    df: DataFrame = pd.read_csv("./zoo.csv")

    indexes: list[int] = get_stable_set(df, B, minimum_degree_selection)

    print("Final stable set size:", str(len(indexes)), "/", str(df.shape[0]))
    print("Time taken to find set:", time.time() - startTime)

    filtered_df = df.iloc[indexes]
    print(filtered_df)
    filtered_df.to_csv("./filtered_zoo.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
