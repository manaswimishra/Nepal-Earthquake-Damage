library(party)
library(caret)
library(factoextra)

dataPath      <- 'clean_data'
train_eq      <-  read.csv(file=paste(dataPath,"train_withdistance.csv",sep="/"))
test_eq       <-  read.csv(file=paste(dataPath,"test_withdistance.csv",sep="/"))

num_cols      <-  c('distance_to_epicentre','plinth_area_sq_ft','height_ft_pre_eq','age_building', 'count_floors_pre_eq')

#selecting specific num cols
num_cols      <-  c('distance_to_epicentre','height_ft_pre_eq','age_building')
#scaling
X.train     <- train_eq[num_cols]
X.test      <- test_eq[num_cols]
#scaling
X.train.scale <- scale(X.train) #scaling train
X.train.mean  <- apply(X.train, 2, mean)
X.train.sd    <- apply(X.train, 2, sd)
X.test.scale  <- scale(X.test, center=X.train.mean, scale=X.train.sd)

#K-means loop function
get.kmeans <- function(k){
  set.seed(1000)
  train.cluster <- kmeans(X.train.scale, k, nstart=sample(50:100, 1, replace=TRUE))
  res <- c(k.Value  = k, cluster.vaf = (train.cluster$betweenss)/(train.cluster$totss),           cluster.size = as.data.frame(train.cluster$size), cluster.centroid= as.data.frame(train.cluster$centers), cluster.class=as.data.frame(train.cluster$cluster), cluster.withiness = train.cluster$tot.withinss)
  return (res)
}
max_k <- 8 # Set maximum cluster 
print (Output.kmeans <- sapply(2:max_k, get.kmeans))

vaf_list <- unlist(Output.kmeans['cluster.vaf',1:max_k-1])
wtn_list <- unlist(Output.kmeans['cluster.withiness',1:max_k-1])
scree.test <-data.frame(2:max_k, vaf_list)
elbow.test <-data.frame(2:max_k, wtn_list)
par(mfrow=c(1, 2))
jpeg('plots/scree_plot.jpg')
plot(scree.test, main="Scree Plot", ylab='Variance Accounted For', type='o',col='blue')
dev.off()
jpeg('plots/elbow_plot.jpg')
plot(elbow.test, main="elbow Plot", ylab='SSE Withinness', type='o',col='red')
dev.off()

library(plotly)
par<-(mfrow=c(1,2))
# cluster =3
X.train$Class <- Output.kmeans[,3]$`cluster.class.train.cluster$cluste`

#p<-plot_ly(x = X.train$distance_to_epicentre,
#        y = X.train$age_building,
#        z = X.train$height_ft_pre_eq,
#       type = "scatter3d",
#       mode = "markers",
#        color = as.factor(X.train$Class)) %>%
#  layout(title = "",
#         scene = list(xaxis = list(title = "Epicentre distance"),
#                      yaxis = list(title = "Building age"),
#                      zaxis = list(title = "Height of Building")))
#htmlwidgets::saveWidget(as_widget(p), "Kmeans_3dplot.html")

set.seed(1000)
k.sel <- 4
train.cluster <- kmeans(X.train.scale, k.sel, nstart=sample(50:100, 1, replace=TRUE))
jpeg('plots/train_fviz.jpg')
fviz_cluster(train.cluster, data = X.train[num_cols], main = "Train Cluster plot")
train.cluster.size <- table(train.cluster$cluster)
dev.off()
X.train$Class <- train.cluster$cluster #unscaling

#aggregate function
print(agg_mean<-aggregate(X.train[,1:4],by=list(X.train$Class),FUN=mean, na.rm=TRUE))

set.seed(10000)
test.cluster <- kmeans(X.test.scale, center=train.cluster$centers,nstart=sample(50:100, 1, replace=TRUE))
X.test$Class <- test.cluster$cluster
test.cluster.size <- table(X.test$Class) #unscaling
jpeg('plots/test_fviz.jpg')
fviz_cluster(test.cluster, data = X.test[num_cols],main = "Test Cluster plot")
dev.off()
#cluster sizes
print(cbind(SizeOfClusterTrain = train.cluster.size, PercentageOfClusterTrain = prop.table(train.cluster.size)*100, SizeOfClusterTest = test.cluster.size,PercentageOfClusterTest = prop.table(test.cluster.size)*100))
#centoids
print(cbind(Train.centriods = train.cluster$centers, Test.centroids=test.cluster$centers))
#vafs
print(cbind(TrainVAFs = train.cluster$betweenss/train.cluster$totss,TestVAFs = test.cluster$betweenss/test.cluster$totss))

X.train$damage_grade_num <- train_eq$damage_grade_num
X.test$damage_grade_num  <- test_eq$damage_grade_num

write.csv(X.train,'output/Kmeans_train.csv',row.names=FALSE)
write.csv(X.test,'output/Kmeans_test.csv',row.names=FALSE)