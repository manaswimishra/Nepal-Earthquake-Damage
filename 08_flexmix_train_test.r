library(parallel)

# Calculate the number of cores
no_cores <- detectCores() - 1

# Initiate cluster
cl <- makeCluster(no_cores)

dataPath   <- 'clean_data'
data       <-  read.csv(file=paste(dataPath,"building_structure_ownership_withdistance_283.csv",sep="/"))
relevant   <-  c('distance_to_epicentre','plinth_area_sq_ft','height_ft_pre_eq','age_building','has_superstructure_timber','count_floors_pre_eq', 'land_surface_condition','other_floor_type', 'ground_floor_type', 'roof_type','damage_grade_num','has_superstructure_mud_mortar_stone','has_superstructure_bamboo', 'has_superstructure_adobe_mud', 'has_superstructure_rc_non_engineered','has_superstructure_rc_engineered', 'has_superstructure_other','count_families', 'has_secondary_use', 'position')

data_part  <- data[, relevant]
data_part  <- data_part[100000:200000, ]

run.flexmix <- function(iter, data=data_part){
  library("flexmix")
  library(caret)
  set.seed(3*iter)
  train_index <- sample(1:nrow(data), 0.7 * nrow(data))
  train       <- data[train_index,]
  test        <- data[-train_index,]
  print('Train')
  print(dim(train))
  print('Test')
  print(dim(test))
  # Build a flexmix model on train set and validate that model on validate set
  damage.2=initFlexmix(damage_grade_num ~ ., data = train, k=2, control = list(minprior = 0), model = FLXMRglm(family = "gaussian"),nrep=50)
  print (table(attributes(damage.2)$cluster))
  p_train=rep(NA,dim(train)[1])
  #damage.2
  p1_train=predict(damage.2)$Comp.1
  p2_train=predict(damage.2)$Comp.2
  p_train[clusters(damage.2)==2]=p2_train[clusters(damage.2)==2]
  p_train[clusters(damage.2)==1]=p1_train[clusters(damage.2)==1]
  predictions_train <- round(p_train, 0)
  
  p_test=rep(NA,dim(test)[1])
  #damage.2
  p1_test=predict(damage.2, newdata=test)$Comp.1
  p2_test=predict(damage.2, newdata=test)$Comp.2
  p_test[clusters(damage.2, newdata=test)==2]=p2_test[clusters(damage.2, newdata=test)==2]
  p_test[clusters(damage.2, newdata=test)==1]=p1_test[clusters(damage.2, newdata=test)==1]
  predictions_test <- round(p_test, 0)
  print(dim(test))
  print (length(predictions_test))

  library(caret)
  conf_train = confusionMatrix(as.factor(train$damage_grade_num), as.factor(predictions_train), mode="prec_recall")
  
  conf_test = confusionMatrix(as.factor(test$damage_grade_num), as.factor(predictions_test), mode="prec_recall")
  
  prec.recall.train <- conf_train$byClass
  prec.recall.test  <- conf_train$byClass
  
  # res <- c(k.Value  = as.numeric(iter),  accuracy.train =conf_train$overall[1], accuracy.test = conf_test$overall[1], 
  #          precision.train = prec.recall.train[,"Precision"], precision.test = prec.recall.test[,"Precision"], 
  #          recall.train = prec.recall.train[,"Recall"], recall.test = prec.recall.test[,"Recall"])
  return (predictions_test)
}

max_k = 2
results <- t(as.data.frame((parSapply(cl, 1:max_k, run.flexmix, data=data_part))))
print ('Result summary of sample models')

write.csv(results,'results_flexmix.csv')
stopCluster(cl)