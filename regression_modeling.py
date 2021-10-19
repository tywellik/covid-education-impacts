import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score, explained_variance_score

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
# get mean engagement index for each district
engagement_index['engagement_index'] = engagement_index.mean(axis=1)
engagement_index = engagement_index[['district_id', 'engagement_index']]
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
print(districts.dtypes)
districts = districts.astype({'pct_black/hispanic': float, 'pct_free/reduced': float, 'pp_total_raw': float})
print('districts')
print(districts)

# merge districts and engagement table
engagement = pd.merge(left = districts, right = engagement, on = 'district_id', how = 'outer')
# TODO: for now just drop string variables. at some point - can I incorporate them?
engagement['pct_black/hispanic'].fillna(value=engagement['pct_black/hispanic'].mean(), inplace=True)
engagement['pct_free/reduced'].fillna(value=engagement['pct_free/reduced'].mean(), inplace=True)
engagement['pp_total_raw'].fillna(value=engagement['pp_total_raw'].mean(), inplace=True)
engagement.drop(columns = ['district_id', 'state', 'locale', 'pct_black/hispanic', 'pct_free/reduced', 'pp_total_raw'], inplace = True)
engagement.dropna(inplace = True)
print('engagement')
print(engagement)

# split into X's and y's
ys = engagement['engagement_index']
xs = engagement.drop(['engagement_index'], axis=1)
xs_np = xs.to_numpy()

# perform feature scaling
scaled_features = StandardScaler().fit_transform(xs_np)

# PCA
pca = PCA(n_components = 20)
scaled_features_pca = pd.DataFrame(pca.fit_transform(scaled_features))
print('scaled_features_pca')
print(scaled_features_pca)

# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features_pca, ys, test_size = 0.30, random_state = 8)

# linear regression modeling
lr = LinearRegression().fit(X_train, y_train)
lr_pred_test = lr.predict(X_test)
lr_pred_train = lr.predict(X_train)

print("training MAPE:")
train_mape = mean_absolute_percentage_error(y_train, lr_pred_train)
print(train_mape)
print("test MAPE: ")
test_mape = mean_absolute_percentage_error(y_test, lr_pred_test)
print(test_mape)

print('\ntraining R2 score:')
train_r2 = r2_score(y_train, lr_pred_train)
print(train_r2)
print('test R2 score:')
test_r2 = r2_score(y_test, lr_pred_test)
print(test_r2)

print('\ntraining explained variance score:')
train_expl_var = explained_variance_score(y_train, lr_pred_train)
print(train_expl_var)
print('test explained variance score:')
test_expl_var = explained_variance_score(y_test, lr_pred_test)
print(test_expl_var)