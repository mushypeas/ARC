import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


train = pd.read_csv('data/mnist_train.csv')

data = train.drop(labels = ["label"],axis = 1)
label = train['label']

pca = PCA(n_components=50)
pca_data = pca.fit_transform(data)
tsne = TSNE(n_components=2, learning_rate='auto', init='random')
tsne_res = tsne.fit_transform(pca_data)

plt.figure(figsize=(16,10))
sns.scatterplot(
  x = tsne_res[:,0],
  y = tsne_res[:,1],
  hue = label,
  palette = sns.hls_palette(10),
  legend = 'full'
)
plt.show()