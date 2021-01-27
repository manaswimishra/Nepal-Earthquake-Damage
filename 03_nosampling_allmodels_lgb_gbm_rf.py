import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import time

start_time = time.time()

train_data = pd.read_csv('clean_data/train_withdistance.csv')
test_data  = pd.read_csv('clean_data/test_withdistance.csv')
train_data = reduce_memory(train_data)
test_data  = reduce_memory(test_data)

#merge data together
num_train  = train_data.shape[0]
data_merge = pd.concat([train_data,test_data],axis=0)
drop_cols  = ['vdcmun_id', 'ward_id', 'district_name', 'district_id', 'damage_grade', 'building_id', 'damage_grade_num', 'count_floors_pre_eq']
data_merge.drop(drop_cols, axis=1, inplace=True)
print ('dropped columns...')

#changed datatype - Our predictors are either bool(int8), float16 or object data type. 'building_id' and 'damage_grade' can be ignored for transformation and feature engineering
boolean_cols = ['has_secondary_use_agriculture','has_secondary_use_hotel', 'has_secondary_use_rental', 'has_secondary_use_institution',
'has_secondary_use_school','has_secondary_use_industry','has_secondary_use_health_post','has_secondary_use_gov_office','has_secondary_use_use_police','has_secondary_use_other','has_superstructure_adobe_mud','has_superstructure_mud_mortar_stone','has_superstructure_stone_flag','has_superstructure_cement_mortar_stone','has_superstructure_mud_mortar_brick','has_superstructure_cement_mortar_brick','has_superstructure_timber','has_superstructure_bamboo','has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered','has_superstructure_other']
data_merge[boolean_cols] = data_merge[boolean_cols].astype('uint8')
numeric_cols             = ['age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq', 'distance_to_epicentre']
data_merge[numeric_cols] = data_merge[numeric_cols].astype('float16')
cat_cols = data_merge.select_dtypes(include=['category', object]).columns
encode_df  = pd.get_dummies(data_merge, columns=cat_cols)
print ('done with one hot encoding')
print ('shape of encode df:', encode_df.shape)

# X-Y
X_train    = encode_df.iloc[0:num_train,]
X_test     = encode_df.iloc[num_train:,]

print('train shape:', X_train.shape)
print('test shape:', X_test.shape)
 
Y_train  = train_data['damage_grade_num'].values
Y_test   = test_data['damage_grade_num'].values
gc.collect()

# train-val splits
train_, val_, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 99, stratify=Y_train)

# Build LGB model
print("LightGBM started...")

pred_train_lgb, pred_test_lgb, model_lgb, evals_result = fit_lgb(train_, y_train, val_, y_val, X_test, X_train)

# save all files
save_file('LightGBM', model_lgb, train_data, test_data, pred_train_lgb, pred_test_lgb, X_train.columns)

print("LightGBM Completed...")

# Build RF model
print("RF started...")
pred_train_rf, pred_test_rf, model_rf = fit_rf(X_train, Y_train, X_test)

# save all files
save_file('RF', model_rf, train_data, test_data, pred_train_rf, pred_test_rf, X_train.columns)

print("RF Completed...")

# Build GBM model
print("GBM started...")
pred_train_gbm, pred_test_gbm, model_gbm = fit_gbm(X_train, Y_train, X_test)

# save all files
save_file('GBM', model_gbm, train_data, test_data, pred_train_gbm, pred_test_gbm, X_train.columns)

print("GBM Completed...")

print("--- %s seconds ---" % (time.time() - start_time))