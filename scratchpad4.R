# In this case I will try to make use of the earlier predictions and will use their "common knowledge"
# to perform new predictions:

csv.list <- dir(pattern = ".csv")[1:19]

data.list <- lapply(csv.list, function(x){
        temp <- as.data.frame(read.csv(x))
})

combined.data <- NULL
for(i in seq_along(data.list)){
        Prediction <- data.list[i][[1]][,2]
        combined.data <- cbind(combined.data,Prediction)
}

row.names(combined.data) <- data.list[1][[1]][,1]

prediction.sum <- apply(combined.data,1,sum)

hist(prediction.sum)

majority.vote <- ifelse(prediction.sum < 9.5, 0,1)

# Let's perform a "majority vote" prediction

prediction.table <- data.frame(PassengerId = row.names(combined.data), 
                               Survived = majority.vote)   
write.csv(prediction.table,paste0("MajorityVote_predictions_4",".csv"), row.names = F)

# The accuracy of this prediction is the same as PCA.svm in scratchpad3: 0.79426

# Can we use unsupervised learning to improve the accuracy?

set.seed(1234)
prediction.kmeans <- kmeans(t(combined.data),4,100)
prediction.kcenters <- data.frame(t(prediction.kmeans$centers))
prediction.kcenters$mean.center <- apply(prediction.kcenters,1,mean)
prediction.kcenters$majority.vote <- ifelse(prediction.kcenters$mean.center < 0.5, 0,1)

# Let's perform another "majority vote" prediction

prediction.table <- data.frame(PassengerId = row.names(prediction.kcenters), 
                               Survived = prediction.kcenters$majority.vote)   
write.csv(prediction.table,paste0("MajorityVote_predictions_4_1",".csv"), row.names = F)

# This is lower accuracy compared to simple averaging: 0.78947

