from k_means import KMEANS

test_data=[[1,1],
[5,5],
[80,7],
[6,6],
[90,5],
[77,10],
[88,90],
[77,80],
[66,100]]

clustering=KMEANS()

clustering.Clustering(test_data,3)

print(clustering.clusters)