library(MASS)
library(caret)
library(nnet)

dataPath      <- 'clean_data'
train_eq      <-  read.csv(file=paste(dataPath,"train_withdistance.csv",sep="/"))
test_eq       <-  read.csv(file=paste(dataPath,"test_withdistance.csv",sep="/"))

relevant      <-  c('distance_to_epicentre','plinth_area_sq_ft','height_ft_pre_eq','age_building','has_superstructure_timber','count_floors_pre_eq', 'land_surface_condition','other_floor_type', 'ground_floor_type', 'roof_type','damage_grade_num','has_superstructure_mud_mortar_stone','has_superstructure_bamboo', 'has_superstructure_adobe_mud', 'has_superstructure_rc_non_engineered','has_superstructure_rc_engineered', 'has_superstructure_other','count_families', 'has_secondary_use', 'position')
num_cols      <-  c('distance_to_epicentre','plinth_area_sq_ft','height_ft_pre_eq','age_building') #

train_eq_part <- train_eq[, relevant]
test_eq_part  <- test_eq[, relevant]

#scaling
X.train.mean  <- apply(train_eq_part[num_cols], 2, mean)
X.train.sd    <- apply(train_eq_part[num_cols], 2, sd)
train_eq_part[num_cols] <- scale(train_eq_part[num_cols], center = TRUE, scale = TRUE) #scaling train
test_eq_part[num_cols]  <- scale(test_eq_part[num_cols], center=X.train.mean, scale=X.train.sd)

train_eq_part$damage_grade_num <- as.factor(train_eq_part$damage_grade_num)
test_eq_part$damage_grade_num  <- as.factor(test_eq_part$damage_grade_num)

# fit a polr logistic model
polr.fit.log  <- polr(damage_grade_num ~. ,data=train_eq_part, method='logistic', Hess=TRUE)
print('model summary')
print(summary(polr.fit.log))

test_eq_part_predClass_ord <- predict(polr.fit.log, newdata = test_eq_part, type="class", se = TRUE)
test_eq_part_predProbs_ord <- predict(polr.fit.log, newdata = test_eq_part, type="probs", se = TRUE)

train_eq_part_predClass_ord <- predict(polr.fit.log, type="class", se = TRUE)

con_train_polr <- confusionMatrix(train_eq_part$damage_grade_num, train_eq_part_predClass_ord, mode="prec_recall")
con_test_polr <- confusionMatrix(test_eq_part$damage_grade_num,test_eq_part_predClass_ord, mode="prec_recall")

print('Train results...')
print(con_train_polr)
print('Test results...')
print(con_test_polr)

train.output.polr <- as.data.frame(c(data.frame(train_eq$building_id),data.frame(train_eq_part$damage_grade_num),data.frame(train_eq_part_predClass_ord), data.frame(polr.fit.log$fitted.values)))
test.output.polr  <- as.data.frame(c(data.frame(test_eq$building_id),data.frame(test_eq_part$damage_grade_num),data.frame(test_eq_part_predClass_ord), data.frame(test_eq_part_predProbs_ord)))

write.csv(train.output.polr,'output/polr_log_train.csv',row.names=FALSE)
write.csv(test.output.polr,'output/polr_log_test.csv',row.names=FALSE)
saveRDS(polr.fit.log, "savedmodels/polr_log_fit.rds")


# fit a multinomial model
multinom.fit  <- multinom(damage_grade_num ~. ,data=train_eq_part)
z             <- summary(multinom.fit)$coefficients/summary(multinom.fit)$standard.errors
print('model summary')
print(summary(multinom.fit))

test_eq_part_predClass_multi <- predict(multinom.fit, newdata = test_eq_part, type="class", se = TRUE)
test_eq_part_predProbs_multi <- predict(multinom.fit, newdata = test_eq_part, type="probs", se = TRUE)

train_eq_part_predClass_multi <- predict(multinom.fit, type="class", se = TRUE)

con_train_multi <- confusionMatrix( train_eq_part$damage_grade_num, train_eq_part_predClass_multi, mode="prec_recall")
con_test_multi  <- confusionMatrix(test_eq_part$damage_grade_num,test_eq_part_predClass_multi, mode="prec_recall")

print('Train results...')
print(con_train_multi)
print('Test results...')
print(con_test_multi)

train.output.multi <- as.data.frame(c(data.frame(train_eq_part$damage_grade_num),data.frame(train_eq_part_predClass_multi), data.frame(multinom.fit$fitted.values)))
test.output.multi  <- as.data.frame(c(data.frame(test_eq_part$damage_grade_num),data.frame(test_eq_part_predClass_multi), data.frame(test_eq_part_predProbs_multi)))

write.csv(train.output.multi,'output/multinom_train.csv',row.names=FALSE)
write.csv(test.output.multi,'output/multinom_test.csv',row.names=FALSE)
saveRDS(multinom.fit, "savedmodels/multinom_fit.rds")
