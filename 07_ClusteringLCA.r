library(poLCA)
library(hashmap)
library(tidyr)
library(ggplot2)

dataPath     <- 'clean_data'
train_eq     <-  read.csv(file=paste(dataPath,"train_withdistance.csv",sep="/"), stringsAsFactors=TRUE)
test_eq      <-  read.csv(file=paste(dataPath,"test_withdistance.csv",sep="/"), stringsAsFactors=TRUE)
train_eq     %>% drop_na()
test_eq      %>% drop_na()

relevant.columns <- c('distance_to_epicentre_factor','height_ft_pre_eq_factor','age_factor','plinch_area_sq_ft_factor', 'land_surface_condition', 'foundation_type', 'ground_floor_type')

transform.num.data <- function(data){
  data$distance_to_epicentre <- as.numeric(data$distance_to_epicentre)
  data$distance_to_epicentre_factor<- cut(x = data$distance_to_epicentre, breaks = c(0, 15, 40, 60, 90, 140, 230))
  data$age_building <- as.numeric(data$age_building)
  data$age_factor<- cut(x = data$age_building, breaks = c(0, 10, 20, 30,100000))
  data$plinth_area_sq_ft <- as.numeric(data$plinth_area_sq_ft)
  data$plinch_area_sq_ft_factor<- cut(x = data$plinth_area_sq_ft, breaks = c(0, 200, 250, 300, 350, 400,600,10000000))
  data$height_ft_pre_eq <- as.numeric(data$height_ft_pre_eq)
  data$height_ft_pre_eq_factor <- cut(x = data$height_ft_pre_eq, breaks = c(5, 10, 15, 20, 25,10000000))
  return(data)
}

train_eq    <- transform.num.data(train_eq)
test_eq     <- transform.num.data(test_eq)

test_eq  <- test_eq[,relevant.columns]
train_eq <- train_eq[,relevant.columns]


convert_manual_factors <- function(data){
  
  for (col in names(data)){
    levels_ <- levels(data[,col])
    H       <- hashmap(levels_, c(1:length(levels_)))
    data[, col] <- as.numeric(factor(data[, col], levels= H$keys(), labels=H$values()))
  }
  return (data)
}

test_eq <- convert_manual_factors(test_eq)
train_eq <- convert_manual_factors(train_eq)

f <- cbind(distance_to_epicentre_factor,age_factor,height_ft_pre_eq_factor,plinch_area_sq_ft_factor, land_surface_condition, foundation_type, ground_floor_type)~1


get.polCA <- function(k){
  set.seed(1000)
  train.cluster <- poLCA(f, train_eq, nclass=k, nrep=10, tol=0.001, verbose = FALSE, graphs= TRUE, probs.start = sample(50:100, 1, replace=TRUE))
  res <- c(k.Value  = as.numeric(k), AIC = train.cluster$aic, BIC =train.cluster$bic)
  return (res)
}
max_k <- 5
results <- sapply(2:max_k, get.polCA)
#print (head(data.frame(t(as.matrix(results)))))

#plot AIC
plot.df <- data.frame(t(as.matrix(results)))
jpeg('plots/lca_aic.jpg')
ggplot(plot.df, aes(k.Value)) +
geom_line(aes(y = AIC), color = "blue") 
dev.off()
jpeg('plots/lca_bic.jpg')
ggplot(plot.df, aes(k.Value)) +
geom_line(aes(y = BIC), color = "red") 
dev.off()


#holdout validation of LCA
#optimal solution for LCA selected K is 3
k <- 3
train.cluster   <- poLCA(f, train_eq, nclass=k, nrep=10, tol=0.001, verbose = FALSE, graphs= TRUE, probs.start = sample(50:100, 1, replace=TRUE))

#plots
jpeg('plots/lca_train_cluster.jpg')
poLCA(f, train_eq, nclass=k, nrep=10, tol=0.001, verbose = FALSE, graphs= TRUE, probs.start = sample(50:100, 1, replace=TRUE))
dev.off()

probs.start     <- train.cluster$probs.start
new.probs.start <- poLCA.reorder(probs.start, c(1, 3, 2))
train.cluster   <- poLCA(f, train_eq, nclass=k, nrep=10, tol=0.001, verbose = FALSE, graphs= TRUE, probs.start = new.probs.start)
probs.start <- poLCA.reorder(train.cluster$probs.start,order(train.cluster$P, decreasing = TRUE))
#plots
jpeg('plots/lca_test_cluster.jpg')
poLCA(f, test_eq, probs.start = probs.start, graphs= TRUE)
dev.off()

test.cluster  <- poLCA(f, test_eq, probs.start = probs.start)
train_eq$PredClass <- train.cluster$predclass
test_eq$PredClass <- test.cluster
print(cbind(AIC = train.cluster$aic, BIC= train.cluster$bic, GSQ = train.cluster$Gsq, ChiSq = train.cluster$Chisq))

saveRDS(train.cluster, "savedmodels/lca_cluster.rds")
write.csv(train_eq,'output/lca_train.csv',row.names=FALSE)
write.csv(test_eq,'output/lca_test.csv',row.names=FALSE)
