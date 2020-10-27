import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(16,6))
iris_data = pd.read_csv('Iris.csv', index_col = 'SepalLengthCm')
df  = iris_data[['SepalLengthCm','PetalWidthCm']]
sns.lineplot(data=df)
