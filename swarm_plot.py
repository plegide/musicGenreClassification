import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd

data = [
    ["ANN", 69.75, 1],
    ["ANN", 75.63, 1],
    ["ANN", 75.59, 1],
    ["ANN", 91.42, 1],
    ["ANN", 83.30, 1],
    ["ANN", 73.21, 1],
    ["ANN", 67.72, 1],
    ["ANN", 84.69, 1],
    ["KNN", 95.46, 1],
    ["KNN", 95.36, 1],
    ["KNN", 95.54, 1],
    ["KNN", 95.50, 1],
    ["KNN", 95.21, 1],
    ["KNN", 95.64, 1],
    ["KNN", 95.54, 1],
    ["SVM", 94.82, 1],
    ["SVM", 94.14, 1],
    ["SVM", 91.96, 1],
    ["SVM", 84.03, 1],
    ["SVM", 94.42, 1],
    ["SVM", 93.69, 1],
    ["SVM", 92.64, 1],
    ["SVM", 89.57, 1],
    ["DT", 93.17, 1],
    ["DT", 93.17, 1],
    ["DT", 94.50, 1],
    ["DT", 94.92, 1],
    ["DT", 95.25, 1],
    ["DT", 94.92, 1],

    #APROX2
    ["ANN", 64.18, 2],
    ["ANN", 61.27, 2],
    ["ANN", 61.72, 2],
    ["ANN", 61.62, 2],
    ["ANN", 59.32, 2],
    ["ANN", 61.74, 2],
    ["ANN", 63.17, 2],
    ["ANN", 59.17, 2],
    
    ["KNN", 73.93, 2],
    ["KNN", 75.15, 2],
    ["KNN", 74.74, 2],
    ["KNN", 75.34, 2],
    ["KNN", 74.98, 2],
    ["KNN", 75.44, 2],
    ["KNN", 74.79, 2],

    ["SVM", 75.39, 2],
    ["SVM", 72.72, 2],
    ["SVM", 64.02, 2],
    ["SVM", 50.14, 2],
    ["SVM", 75.37, 2],
    ["SVM", 73.29, 2],
    ["SVM", 66.43, 2],
    ["SVM", 40.49, 2],

    ["DT", 61.70, 2],
    ["DT", 72.09, 2],
    ["DT", 71.49, 2],
    ["DT", 73.50, 2],
    ["DT", 74.05, 2],
    ["DT", 74.39, 2],
    #APROX3

    ["ANN", 72.91, 3],
    ["ANN", 69.36, 3],
    ["ANN", 69.03, 3],
    ["ANN", 68.99, 3],
    ["ANN", 66.70, 3],
    ["ANN", 66.07, 3],
    ["ANN", 70.74, 3],
    ["ANN", 66.37, 3],

    ["KNN", 83.54, 3],
    ["KNN", 83.39, 3],
    ["KNN", 82.80, 3],
    ["KNN", 83.48, 3],
    ["KNN", 82.91, 3],
    ["KNN", 82.53, 3],
    ["KNN", 82.68, 3],

    ["SVM", 77.27, 3],
    ["SVM", 72.28, 3],
    ["SVM", 77.21, 3],
    ["SVM", 2.37, 3],
    ["SVM", 81.83, 3],
    ["SVM", 73.57, 3],
    ["SVM", 79.64, 3],
    ["SVM", 2.33, 3],

    ["DT", 46.44, 3],
    ["DT", 65.91, 3],
    ["DT", 67.76, 3],
    ["DT", 70.00, 3],
    ["DT", 72.24, 3],
    ["DT", 75.31, 3],

    ["ANN", 82.77, 4],
    ["ANN", 80.97, 4],
    ["ANN", 80.77, 4],
    ["ANN", 80.45, 4],
    ["ANN", 79.54, 4],
    ["ANN", 80.57, 4],
    ["ANN", 81.98, 4],
    ["ANN", 78.67, 4],

    ["KNN", 78.99, 4],
    ["KNN", 78.68, 4],
    ["KNN", 77.86, 4],
    ["KNN", 77.49, 4],
    ["KNN", 77.20, 4],
    ["KNN", 76.77, 4],
    ["KNN", 76.83, 4],

    ["SVM", 66.96, 4],
    ["SVM", 60.0, 4],
    ["SVM", 66.95, 4],
    ["SVM", 1.30, 4],
    ["SVM", 75.20, 4],
    ["SVM", 61.98, 4],
    ["SVM", 71.00, 4],
    ["SVM", 3.32, 4],

    ["DT", 29.79, 4],
    ["DT", 41.67, 4],
    ["DT", 47.57, 4],
    ["DT", 54.44, 4],
    ["DT", 55.89, 4],
    ["DT", 60.07, 4],
]

df = pd.DataFrame(data, columns=['Modelo de aprendizaje automático', 'Precisión', 'Aproximación'])

for i in range(1,5):
    aprox_data = df[df['Aproximación'] == i]
# Create a combined swarmplot with boxes
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Modelo de aprendizaje automático', y='Precisión', data=aprox_data, hue="Modelo de aprendizaje automático", palette='muted', linewidth=1)  # Swarmplot
    sns.boxplot(x='Modelo de aprendizaje automático', y='Precisión', data=aprox_data, width=0.5, fill=False, color='black')  # Boxplot
    plt.title('Precisión de los modelos de aprendizaje automático en la aproximación ' + str(i))
    plt.xlabel('Modelo de aprendizaje automático')
    plt.ylabel('Precisión (%)')
    plt.savefig('graficas/aprox' + str(i) + '.png')
