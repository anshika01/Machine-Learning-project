import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing

# scree plot
destinations = pd.read_csv("destinations.csv")
components = 10
pca = PCA(n_components=components)
scaler = preprocessing.StandardScaler()
destinations = scaler.fit_transform(destinations)
pca.fit(destinations)
variance= pca.explained_variance_ratio_
print variance
cumulative_explained_variance=np.cumsum(variance)
plt.plot(cumulative_explained_variance)
plt.xlabel('components number', fontsize=14)
plt.ylabel('cumulative explained variance', fontsize=14)
plt.show()

# biplot
destinations = pd.read_csv("destinations.csv")
components = 3
pca = PCA(n_components=components)
#scaler = preprocessing.StandardScaler()
#destinations = scaler.fit_transform(destinations)
pca.fit(destinations)
df_pca = pca.fit_transform(destinations)

## project data into PC space
# 0,1 denote PC1 and PC2; change values for other PCs
xvector = pca.components_[0]
yvector = pca.components_[1]

xs = df_pca[:, 0]
ys = df_pca[:, 1]

## visualize projections

for i in range(len(xvector)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i] * max(xs), yvector[i] * max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvector[i] * max(xs) * 1.2, yvector[i] * max(ys) * 1.2,
             list(destinations.columns.values)[i],
             color='r')

for i in range(len(xs)):
    # circles project documents (ie rows from csv) as points onto PC axes
    plt.plot(xs[i], ys[i], 'bo')
    plt.text(xs[i] * 1.2, ys[i] * 1.2, list(destinations.index)[i], color='b')

plt.show()