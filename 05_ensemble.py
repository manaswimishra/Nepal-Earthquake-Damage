import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import glob
import pickle

#path to declare
path = 'output/'

#read train files

train_lgb       = pd.read_csv(path+'LightGBM_train.csv')
train_multinom  = pd.read_csv(path+'multinom_train.csv')
train_polr      = pd.read_csv(path+'polr_log_train.csv')
train_rf        = pd.read_csv(path+'RF_train.csv')
train_gbm       = pd.read_csv(path+'GBM_train.csv')

#read test files

test_lgb       = pd.read_csv(path+'LightGBM_test.csv')
test_multinom  = pd.read_csv(path+'multinom_test.csv')
test_polr      = pd.read_csv(path+'polr_log_test.csv')
test_rf        = pd.read_csv(path+'RF_test.csv')
test_gbm       = pd.read_csv(path+'GBM_test.csv')

def get_clean_report(lgb, multinom, polr, rf, gbm, word):
    lgb.rename(columns={'0':'LGB_Grade1', '1': 'LGB_Grade2', '2':'LGB_Grade3', '3':'LGB_Grade4'}, inplace=True)
    multinom.rename(columns={'X0':'Multinom_Grade1', 'X1': 'Multinom_Grade2', 'X2':'Multinom_Grade3', 'X3':'Multinom_Grade4', word+'_eq.building_id':'building_id', word+'_eq_part.damage_grade_num':'damage_grade_num',
       word+'_eq_part_predClass_multi':'damage_pred'}, inplace=True)
    polr.rename(columns={'X0':'OrdLogistic_Grade1', 'X1': 'OrdLogistic_Grade2', 'X2':'OrdLogistic_Grade3', 'X3':'OrdLogistic_Grade4', word+'_eq.building_id':'building_id', word+'_eq_part.damage_grade_num':'damage_grade_num',
       word+'_eq_part_predClass_ord':'damage_pred'}, inplace=True)
    rf.rename(columns={'0':'RF_Grade1', '1': 'RF_Grade2', '2':'RF_Grade3', '3':'RF_Grade4'}, inplace=True)
    gbm.rename(columns={'0':'GBM_Grade1', '1': 'GBM_Grade2', '2':'GBM_Grade3', '3':'GBM_Grade4'}, inplace=True)
    print ('results for ', word)
    print ("accuracy of multinomial model is : {}". format(accuracy_score(multinom['damage_grade_num'], multinom['damage_pred'])))
    print ("accuracy of ordinal logistics model is : {}". format(accuracy_score(polr['damage_grade_num'], polr['damage_pred'])))
    print ("accuracy of LightGBM model is : {}". format(accuracy_score(lgb['damage_grade_num'], lgb['damage_pred'])))
    print ("accuracy of RandomForest model is : {}". format(accuracy_score(rf['damage_grade_num'], rf['damage_pred'])))
    print ("accuracy of RandomForest model is : {}". format(accuracy_score(gbm['damage_grade_num'], gbm['damage_pred'])))
    
def get_probs(lgb, multinom, polr, rf, gbm):
    drop_cols = ['damage_grade_num', 'damage_pred']
    lgb_probs      = lgb.drop(drop_cols, axis=1)
    multinom_probs = multinom.drop(drop_cols, axis=1)
    polr_probs     = polr.drop(drop_cols, axis=1)
    rf_probs       = rf.drop(drop_cols, axis=1)
    gbm_probs      = gbm.drop(drop_cols, axis=1)
    
    ensemble = pd.concat([lgb_probs, multinom_probs, polr_probs, rf_probs, gbm_probs], axis=1)
    return (ensemble)
    
print(get_clean_report(train_lgb, train_multinom, train_polr, train_rf, train_gbm, "train"))
print(get_clean_report(test_lgb, test_multinom, test_polr, test_rf, test_gbm, "test"))

#get data ready for ensemble
train_ensemble = get_probs(train_lgb, train_multinom, train_polr, train_rf, train_gbm)
test_ensemble  = get_probs(test_lgb, test_multinom, test_polr, test_rf, test_gbm)

X_train = train_ensemble
X_test  = test_ensemble
Y_train = train_lgb['damage_grade_num']
Y_test  = test_lgb['damage_grade_num']

#run xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
seed = 0
test_size = 0.3

model = XGBClassifier()
model.fit(X_train, Y_train)

y_pred_class_test  = model.predict(X_test)
y_pred_class_train = model.predict(X_train)
accuracy = accuracy_score(Y_test, y_pred_class_test)
print("Accuracy Test: %.2f%%" % (accuracy * 100.0))
print (confusion_matrix(Y_test, y_pred_class_test))
print (classification_report(Y_test, y_pred_class_test))

accuracy = accuracy_score(Y_train, y_pred_class_train)
print("Accuracy Train: %.2f%%" % (accuracy * 100.0))
print (confusion_matrix(Y_train, y_pred_class_train))
print (classification_report(Y_train, y_pred_class_train))

#save models
file = open('savedmodels/model_ensemble', 'wb') 
pickle.dump(model, file)
file.close()
