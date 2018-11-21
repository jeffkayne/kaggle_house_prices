from sklearn.preprocessing import LabelEncoder
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import os

# display all columns when viewing obj_df.head()
pd.set_option('display.max_columns', None)

# load in test & train data
train_df = pd.read_csv('C:/Users/Jeffrey/Documents/home_projects/houses_all/train.csv')
test_df = pd.read_csv('C:/Users/Jeffrey/Documents/home_projects/houses_all/test.csv')

# set target variable from train_df
target = train_df['SalePrice']

# verify columns in train and test sets are the same
train_cols = train_df.columns
test_cols = test_df.columns
diff_cols = []
for col in train_cols:
    for col_test in test_cols:
        if col_test != col:
            if col_test == test_cols[-1]:
                diff_cols.append(col)
                break
            continue
        else:
            break
print('Different columns: ', diff_cols)

# Create a merged DF containing all features from both test and train, appart from SalePrice.
merged_df = pd.concat([train_df.loc[:, train_df.columns != 'SalePrice'], test_df])
merged_division = len(train_df)

# Divide data in to categorical and quantitative columns ** ACTUALLY COLUMNS ONLY DIVIDED IN TO NUMERICAL AND NON NUMERICAL **
cat_features = []
quant_features = []
for col in merged_df:
    if merged_df[col].dtype == object:
        cat_features.append(col)
        # print(col, ': ', train_dfbis[col].unique())
    elif merged_df[col].dtype == bool:
        merged_df.loc[:, col] = merged_df[col].astype(np.int64)
        quant_features.append(col)
    else:
        quant_features.append(col)

# Format quant_features
merged_df[quant_features] = merged_df[quant_features].fillna(0)

# label encode non numerical data
lb_make = LabelEncoder()
ohe_cols = []
for col in cat_features:
    if col == 'fullVisitorId':
        continue
    if merged_df[col].nunique() < 50:
        ohe_cols.append(col)
        continue
    merged_df[col] = lb_make.fit_transform(merged_df[col].astype(str))
merged_df = pd.get_dummies(merged_df, columns=ohe_cols)

# Reconstruct train and test data frames from modified merged data set
train_df_mod = merged_df[:merged_division]
test_df_mod = merged_df[merged_division:]

# validate performance on training set only
y = target  # supposing that column indexing has not been modified
X = train_df_mod

# Use train test split
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.3, random_state=1)

# Random forrest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_val_predictions)
print("Validation MAE for Random Forest Model: ", rf_val_mae)

model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=720, max_bin=55, bagging_fraction=0.8, bagging_freq=5,
                              feature_fraction=0.2319, feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11)


# Predictions on test set
rf_model.fit(X, y)
X_new = test_df_mod
rf_val_predictions = rf_model.predict(X_new)
sale_price_predict = pd.Series(rf_val_predictions, name='SalePrice', index=X_new.index)
id = X_new['Id']
submission_df = pd.concat([id, sale_price_predict], axis=1)
submission_df.head()

submission_df.to_csv('submission.csv', index=False)
