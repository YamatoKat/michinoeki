import pandas as pd 
import matplotlib.pyplot as plt
from pandas.tools import plotting
from sklearn.cluster import KMeans # K-means
from sklearn.decomposition import PCA
import numpy as np



df = pd.read_csv("michinoeki.csv")
df.head() #データ確認

plotting.scatter_matrix(df[df.columns[1:]], figsize=(6,6), alpha=0.8, diagonal='kde') 
plt.show() #グラフを図示



# 3つのグループに分割 (メルセンヌツイスターの乱数の種を 10 とする)
kmeans_model = KMeans(n_clusters=12, random_state=10).fit(df.iloc[:, 1:])
# 分類結果のラベルを取得
labels = kmeans_model.labels_
# 分類結果を確認
labels


## 分類結果を図示
color_codes = {0:'#800000', 1:'#0000FF', 2:'#000080',3:'#008080', 4:'#008000', 5:'#00FF00',6:'#00FFFF', 7:'#FFFF00', 8:'#FF0000',9:'#FF00FF', 10:'#808000',11:'#800080'}
# サンプル毎に色を与える。
colors = [color_codes[x] for x in labels]
# 色分けした Scatter Matrix を描く。
plotting.scatter_matrix(df[df.columns[1:]], figsize=(6,6), color=colors, alpha=0.8, diagonal='kde')   #データのプロット
plt.show()


#labelsをdfの列に追加
df['label'] = labels
# CSV ファイル (employee.csv) として出力
df.to_csv("employee.csv")


#ここまではK-meansでグラフが作成可能

########
import numpy as np

#labelsごとに、TSP
x=df[df.columns[1:2]] #lat
y=df[df.columns[2:3]] #lon

distance_matrix = np.sqrt((x[:, np.newaxis] - x[np.newaxis, :]) ** 2 + (y[:, np.newaxis] - y[np.newaxis, :]) ** 2)
