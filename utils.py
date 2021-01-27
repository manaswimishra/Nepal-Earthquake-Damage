#!/usr/bin/env python
# coding: utf-8
#Importing necessary Libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gc

#missing value computation
def cal_missing_val(df):
    data_dict = {}
    for col in df.columns:
        data_dict[col] = (df[col].isnull().sum()/df.shape[0])*100
    return pd.DataFrame.from_dict(data_dict, orient='index', columns=['MissingValueInPercentage'])

def replace_un_string(val_s):
    return (val_s.lower().replace('-', '_').replace(' ', '_').replace('/', '_OR_'))

## Reducing memory
## Function to reduce the DF size
def reduce_memory(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def fit_lgb(train_X, train_y, val_X, val_y, test_X, train_all):
    params = params = {
        "objective"  :'multiclass',
        "metric"     : 'multi_logloss', 
        "num_leaves" : 60,
        "learning_rate" : 0.004, 
        "feature_fraction" : 0.8,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 101,
        'boosting_type' : 'gbdt', 
        'scoring': 'roc_auc',
        'num_class': 4
         }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval   = lgb.Dataset(val_X, label=val_y, reference=lgtrain)
    evals_result = {}
    model_lgb   = lgb.train(params, lgtrain, 5000, 
                      valid_sets  =[lgtrain, lgval], 
                      early_stopping_rounds=1000, 
                      verbose_eval=150, 
                      evals_result=evals_result)
    
    pred_train_y = model_lgb.predict(train_all, num_iteration=model_lgb.best_iteration)
    pred_test_y  = model_lgb.predict(test_X, num_iteration=model_lgb.best_iteration)
    return  (pred_train_y, pred_test_y, model_lgb, evals_result)

def fit_rf(X_train, Y_train, X_test):
    #define randomforest model parameters - Picked best.estimator_ from Aalok's GridSearchCV code
    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=35, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=10, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=160,
                       n_jobs=None, oob_score=False, random_state=1, verbose=0,
                       warm_start=False)


    #fit random forest model on whole train set
    rf.fit(X_train, Y_train)
    pred_test_y   = rf.predict_proba(X_test)
    pred_train_y  = rf.predict_proba(X_train)
    
    return (pred_train_y, pred_test_y, rf)

def fit_gbm(X_train, Y_train, X_test):
    gb_clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.04, max_features=8, max_depth=6, random_state=0)
    gb_clf.fit(X_train, Y_train)
    pred_test_y   = gb_clf.predict_proba(X_test)
    pred_train_y  = gb_clf.predict_proba(X_train)
    
    return (pred_train_y, pred_test_y, gb_clf)

#'LightGBM', model_lgb, X_train, X_test, Y_train, Y_test, pred_train_lgb, pred_test_lgb, X_train.columns
#feature_importances_

def save_file(name, model, train, test, pred_train, pred_test, feature_names):
    #save important features by plot and csvs
    if 'LightGBM' in name:
      feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':feature_names})
    else:
      feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':feature_names})
      
    plt.figure(figsize=(40, 20))
    sns.set(font_scale = 3)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:15])
    plt.title(name+' Features')
    plt.tight_layout()
    plt.savefig('plots/'+name+'_imp_features.png')
    feature_imp.to_csv('model_report/'+name+'_imp_features.csv')
    
    #save models
    file = open('savedmodels/model_'+name, 'wb') 
    pickle.dump(model, file)
    file.close()
    
    #confusion matrix and classification reports etc
    
    train['damage_pred'] = np.argmax(np.array(pred_train), axis=1)
    test['damage_pred']  = np.argmax(np.array(pred_test), axis=1)
    confusion_matrix_train = pd.crosstab(train['damage_grade_num'], train['damage_pred'], rownames=['Actual'], colnames=   ['Predicted'])
    print ('Train',confusion_matrix_train)

    confusion_matrix_test = pd.crosstab(test['damage_grade_num'], test['damage_pred'], rownames=['Actual'], colnames=['Predicted'])
    print ('Test', confusion_matrix_test)

    clsf_report_train = pd.DataFrame(classification_report(y_true = train['damage_grade_num'], y_pred = train['damage_pred'], output_dict=True)).transpose()
    clsf_report_train.to_csv('model_report/'+name+'_train_report.csv', index= True)

    clsf_report_test = pd.DataFrame(classification_report(y_true = test['damage_grade_num'], y_pred = test['damage_pred'], output_dict=True)).transpose()
    clsf_report_test.to_csv('model_report/'+name+'_test_report.csv', index= True)

    saveTrain = pd.concat([train[['damage_grade_num','damage_pred']], pd.DataFrame(pred_train)], axis=1)
    saveTest  = pd.concat([test[['damage_grade_num','damage_pred']], pd.DataFrame(pred_test)], axis=1)

    saveTrain.to_csv('output/'+name+'_train.csv', index=False)
    saveTest.to_csv('output/'+name+'_test.csv', index=False)
    print ('Test score', accuracy_score(test['damage_grade_num'], test['damage_pred']))
    print ('Train score', accuracy_score(train['damage_grade_num'], train['damage_pred']))


    
    