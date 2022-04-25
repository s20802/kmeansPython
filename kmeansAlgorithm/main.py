import pandas as pd


def read_csv():

    file = pd.read_csv("Data/iris_cluster_data.txt")
    print(file)


read_csv()
