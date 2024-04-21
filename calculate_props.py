import pandas as pd
import statistics

aprox5_data = pd.read_csv('datasets/aprox5/aprox5.2.data', sep=',')
groups = aprox5_data.groupby(aprox5_data.iloc[:, -1])

for i in range(3, 12):
    print(f"Column {i} Statistics by Last Column Value:")
    for name, group in groups:
        pdMean = group.iloc[:, i].mean()
        pdStd = group.iloc[:, i].std()
        pdMin = group.iloc[:, i].min()
        pdMax = group.iloc[:, i].max()
        print(f"Genre {name}: Mean = {pdMean}, Std = {pdStd}, Min = {pdMin}, Max = {pdMax}")