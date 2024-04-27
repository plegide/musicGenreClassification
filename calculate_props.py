import pandas as pd
import statistics

aprox5_data = pd.read_csv('datasets/aprox6/aprox6.data', sep=',')
groups = aprox5_data.groupby(aprox5_data.iloc[:, -1])

for i in range(3, 10):
    print(f"Column {i} Statistics by Last Column Value:")
    for name, group in groups:
        pdMean = round(group.iloc[:, i].mean(), 2)
        pdStd = round(group.iloc[:, i].std(), 2)
        pdMin = round(group.iloc[:, i].min(), 2)
        pdMax = round(group.iloc[:, i].max(), 2)
        print(f"""
            {name}                &{pdMean} &{pdStd}  &{pdMin}  &{pdMax}\\\ \hline
        """)