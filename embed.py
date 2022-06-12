import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# train
train = pd.read_csv('data/embed_train.csv')

label = train['label']
data = train.drop(labels = ["label"],axis = 1)
data += 128
pd.merge(label,data, right_index=True,left_index=True).to_csv('out.csv')
# pca = PCA(n_components=50)
# pca_data = pca.fit_transform(data)
# tsne = TSNE(n_components=2, learning_rate='auto', init='random')
# tsne_res = tsne.fit_transform(pca_data)

# plt.figure(figsize=(16,10))
# sns.scatterplot(
#   x = data['x'],
#   y = data['y'],
#   hue = label,
#   palette = sns.hls_palette(10),
#   legend = 'full'
# )
# plt.show()


# # test
# test = pd.read_csv('data/embed_test.csv')

# data = test.drop(labels = ["label"],axis = 1)
# label = test['label']

# # pca = PCA(n_components=50)
# # pca_data = pca.fit_transform(data)
# # tsne = TSNE(n_components=2, learning_rate='auto', init='random')
# # tsne_res = tsne.fit_transform(pca_data)

# plt.figure(figsize=(16,10))
# sns.scatterplot(
#   x = data['x'],
#   y = data['y'],
#   hue = label,
#   palette = sns.hls_palette(10),
#   legend = 'full'
# )
# plt.show()
