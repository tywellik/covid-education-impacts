import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# read in products
products = pd.read_csv('data/products_info.csv')
product_ids = products['LP ID'].to_list()

# read in engagement data
# TODO: come back and maybe do some separate groupings by lp id (i.e., products). look at sectors, essential function
engagement = pd.DataFrame()
directory = 'data/engagement_data/'
for filename in os.listdir(directory):
    district_id = filename.split('.')[0]
    dist_df = pd.read_csv(directory + filename)
    dist_df['district_id'] = district_id
    engagement=pd.concat([engagement, dist_df], axis=0)
# update datatypes
engagement['time'] = pd.to_datetime(engagement['time'])
engagement['district_id'] = engagement['district_id'].astype(int)
# get day of week for each date and filter out weekends
engagement['dayOfWeek'] = engagement['time'].dt.day_name()
engagement = engagement[engagement['dayOfWeek'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
del engagement['dayOfWeek']
# also filter out summer and winter breaks
engagement = engagement[engagement['time'] >= pd.to_datetime('2020-01-06')]
engagement = engagement[(engagement['time'] <= pd.to_datetime('2020-06-05')) | (engagement['time'] >= pd.to_datetime('2020-08-17'))]
engagement = engagement[engagement['time'] <= pd.to_datetime('2020-12-18')]
# filter engagement for product ids in the product df
engagement = engagement[engagement['lp_id'].isin(product_ids)]

# pivot engagement data
# pivot engagement_index
engagement_index = pd.pivot_table(engagement, values='engagement_index', index = ['district_id'], columns = ['lp_id'], aggfunc = 'mean')
engagement_index.columns = engagement_index.columns.astype(int).astype(str)
engagement_index = engagement_index.add_prefix('eng_ind_')
engagement_index.reset_index(drop = False, inplace = True)
engagement_index.fillna(0, inplace = True)
# pivot pct_access
engagement_access = pd.pivot_table(engagement, values='pct_access', index = ['district_id'], columns = ['lp_id'], aggfunc = 'mean')
engagement_access.columns = engagement_access.columns.astype(int).astype(str)
engagement_access = engagement_access.add_prefix('pct_acc_')
engagement_access.reset_index(drop = False, inplace = True)
engagement_access.fillna(0, inplace = True)
# merge pivoted engagement tables
engagement = pd.merge(left = engagement_index, right = engagement_access, on = 'district_id', how = 'outer')
engagement = engagement.groupby('district_id').mean()

# read in districts data
districts = pd.read_csv('data/districts_info.csv')
del districts['county_connections_ratio']
# get values from the ranges
districts['pct_black/hispanic'] = districts['pct_black/hispanic'].str.strip('[]')
districts['pct_free/reduced'] = districts['pct_free/reduced'].str.strip('[]')
districts['pp_total_raw'] = districts['pp_total_raw'].str.strip('[]')
districts['pct_black/hispanic'] = districts['pct_black/hispanic'].str.split(', ', expand = True)[1]
districts['pct_free/reduced'] = districts['pct_free/reduced'].str.split(', ', expand = True)[1]
districts['pp_total_raw'] = districts['pp_total_raw'].str.split(', ', expand = True)[1]
print('districts')
print(districts)

# merge districts and engagement table
engagement = pd.merge(left = districts, right = engagement, on = 'district_id', how = 'outer')
# TODO: for now just drop string variables. at some point - can I incorporate them?
# TODO: also dropped 'pct_black/hispanic', 'pct_free/reduced', 'pp_total_raw' because of so many null values...
engagement.drop(columns = ['district_id', 'state', 'locale', 'pct_black/hispanic', 'pct_free/reduced', 'pp_total_raw'], inplace = True)
engagement.dropna(inplace = True)
print('engagement')
print(engagement)
engagement_np = engagement.to_numpy()

# perform feature scaling
scaled_features = StandardScaler().fit_transform(engagement_np)

# TODO: maybe PCA??
pca = PCA(n_components = 20)
scaled_features_pca = pca.fit_transform(scaled_features)
print('scaled_features_pca')
print(pd.DataFrame(scaled_features_pca))

# run kmeans iteratively with different values of k and record the SSE for each k 
print('run kmeans iteratively with different values of k')

kmeans_args = {"init": "random",
               "n_init": 50,
               "max_iter": 300,
               "random_state": 42,}
sse = []
for k in range(1, 12):
    kmeans = KMeans(n_clusters=k, **kmeans_args)
    kmeans.fit(scaled_features_pca)
    sse.append(kmeans.inertia_)

# plot k vs. SSE to see if we can identify an "elbow"
plt.style.use("fivethirtyeight")
plt.figure(figsize=(12,10))
plt.plot(range(1, 12), sse)
plt.xticks(range(1, 12, 2))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.savefig('figures/kmeans_elbow.png')

# elbow method didn't really work... use 3 clusters??
print('run through 200 iterations of k means and use the lowest cost model')
kmeans = KMeans(init = "random",
                n_clusters = 4,
                n_init = 200,
                max_iter = 300)

kmeans.fit(scaled_features_pca)
# print(kmeans.inertia_)
# print(kmeans.cluster_centers_)
# print(kmeans.n_iter_)
# print(kmeans.labels_[:5])

# PCA to 2 dimensions to visualize clustering
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(engagement_np)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
intermediaryDf = pd.DataFrame(kmeans.labels_)
finalDf = pd.concat([principalDf, intermediaryDf], axis = 1).rename(columns = {0: 'cluster label'})

# plot clusters
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_yscale('log')
ax.set_title('2 component PCA view of clusters', fontsize = 20)
targets = [0, 1, 2]
# colors = ['r', 'g', 'b', 'c', 'm', 'y']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['cluster label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.savefig('figures/cluster_pca.png')