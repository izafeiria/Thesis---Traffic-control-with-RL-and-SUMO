import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\zafeiria\\Desktop\\QLearning\\analysis\\features.csv', sep=";")
df = pd.DataFrame(data)
fig, ax = plt.subplots()
sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.4f', 
            cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
plt.savefig('corMatrixFeatures.png', bbox_inches='tight', pad_inches=0.0)
