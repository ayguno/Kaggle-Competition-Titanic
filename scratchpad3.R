# Next, I can go back to perform: 
# - a few more feature engineering steps to add some features I was able to extract previously
# - repeat PCA 
# - fit PCA.svm again
# - Test whether adding a few more features before PCA can improve the accuracy 

library(dplyr);library(ggplot2); library(caret)

training <- read.csv("train.csv")
final_testing <- read.csv("test.csv")


training <- dplyr::select(training, -PassengerId)
training$Survived <- factor(training$Survived)

# remove some features:

training <- dplyr::select(training,-Cabin, - Embarked, -Age)
final_testing <- dplyr::select(final_testing,-Cabin, - Embarked, -Age)

summary(training)
summary(final_testing)

# There is one missing value in the Fare feature in the testing set, thus we need to find a way to impute by using the 
# training set

lmFit1 <- lm(Fare ~ factor(Pclass)*factor(Sex), data = training)

summary(lmFit1)

par(mfrow= c(2,2))
plot(lmFit1)[1:4]

# lmFit1 Fair model to impute Fare in the testing set

sum(is.na(training$Fare)) # No missing values in the training set
sum(is.na(final_testing$Fare)) # One missing value in the test set

final_testing$Fare[is.na(final_testing$Fare)] <- predict(lmFit1, newdata = final_testing[is.na(final_testing$Fare),])

sum(is.na(final_testing$Fare)) # Imputed

# Final check confirms that there is no remaining missing data in either of the data sets
apply(is.na(training),2,sum);apply(is.na(final_testing),2,sum)

# Convert Sex to dummy variable
training$Female <- ifelse(training$Sex == "female", 1,0)
final_testing$Female <- ifelse(final_testing$Sex == "female", 1,0)

training <- dplyr::select(training, - Sex)
final_testing <- dplyr::select(final_testing, -Sex)


# Ticket
Ticket <- toupper(training$Ticket)
# Better to remove anything left of the last white space
w <- grep(" ", Ticket)
last.space<- sapply(gregexpr(" ",Ticket[w]), function(y){
        max(y[1])
})

Ticket[w] <- substring(Ticket[w],last.space+1)

Ticket <- gsub(" ","", Ticket)
Ticket <- gsub("[A-Z]","",Ticket)
Ticket <- gsub("\\.","",Ticket)
Ticket <- gsub("\\/","", Ticket)
Ticket <- as.numeric(Ticket)
Ticket[is.na(Ticket)] = 0
Ticket <- as.numeric(Ticket)


plot(x=log10(Ticket), y = training$Fare, col = ifelse(training$Survived == 0, "red","navy"))
plot(x=log10(Ticket), y = training$SibSp, col = ifelse(training$Survived == 0, "red","navy"))
plot(x=log10(Ticket), y = training$Parch, col = ifelse(training$Survived == 0, "red","navy"))

# Remove original Ticket and add new Ticket feature

# Feature Engineering function
num.Ticket <- function(x){
        Ticket <- toupper(x$Ticket)
        # Better to remove anything left of the last white space
        w <- grep(" ", Ticket)
        last.space<- sapply(gregexpr(" ",Ticket[w]), function(y){
                max(y[1])
        })
        Ticket[w] <- substring(Ticket[w],last.space+1)
        
        Ticket <- gsub(" ","", Ticket)
        Ticket <- gsub("[A-Z]","",Ticket)
        Ticket <- gsub("\\.","",Ticket)
        Ticket <- gsub("\\/","", Ticket)
        Ticket <- as.numeric(Ticket)
        Ticket <- as.numeric(Ticket)
        return(log10(Ticket))
}

training$num.Ticket <- num.Ticket(training)
final_testing$num.Ticket <- num.Ticket(final_testing)


sum(is.na(training$num.Ticket )) # 4 Missing values introduced
sum(is.na(final_testing$num.Ticket)) # No missing values

# remove the Ticket feature

training <- dplyr::select(training,-Ticket)
final_testing <- dplyr::select(final_testing,-Ticket)

# Title feature from Name:

# Feature engineering function:
Titles.func <- function(x){
        Pass.Names <- x$Name
        
        first.comma<- sapply(gregexpr(",",Pass.Names), function(y){
                y[1][1]
        })
        first.dot <- sapply(gregexpr("\\.",Pass.Names), function(y){
                y[1][1]
        })
        
        Titles <- substr(Pass.Names,first.comma+2,first.dot-1)  
        return(Titles)
        }

Titles <- Titles.func(training)

qplot(x = training$Fare[training$Female == 0], y = factor(Titles[training$Female == 0]), data = training[training$Female == 0,], color = Survived, alpha = I(0.2))+
        theme_bw()

# We notice that "Master" title explains quite a few of the survived males. This would be a useful feature to add into the data sets

training$Titles.Master <- ifelse(Titles == "Master",1,0)
final_testing$Titles.Master <- ifelse(Titles.func(final_testing) == "Master",1,0)

# Remove the original Name feature:

training <- dplyr::select(training,-Name)
final_testing <- dplyr::select(final_testing,-Name)

apply(is.na(training),2,sum)
apply(is.na(final_testing),2,sum)

training <- training[complete.cases(training),]

##############
#Perform PCA
##############

# Dimension reduction with the training set

prePCA <- preProcess(training[,-1],method = "pca")
PCAtraining <- predict(prePCA,newdata = training)

qplot(x = PC1, y = PC2, data = PCAtraining, color = Survived, alpha = I(0.3))+theme_bw()+
        scale_color_manual(values = c("red","navy"))

set.seed(1234)
PCA.svm <- train(Survived ~ ., data = PCAtraining,method = "svmRadial",
                 trControl= trainControl(method = "boot632"), verbose =F)

# Let's perform a prediction:

PCAtesting <- predict(prePCA,newdata = final_testing)


prediction.table <- data.frame(PassengerId = final_testing$PassengerId, 
                               Survived = predict(PCA.svm,PCAtesting))   
write.csv(prediction.table,paste0("PCA_predictions_3",".csv"), row.names = F)

# Great work! These two new features alone have increased the testing accuracy from 0.77 to 0.79426!
# Next, try to perform predictions with other models to see if we can further improve this accuracy with the 
# existing training set

set.seed(1234)
PCA.amdai <- train(Survived ~ ., data = PCAtraining,method = "amdai",
                   trControl= trainControl(method = "boot632"))


set.seed(1234)
PCA.rf <- train(Survived ~ ., data = PCAtraining,method = "rf",
                trControl= trainControl(method = "boot632"))


set.seed(1234)
PCA.gbm <- train(Survived ~ ., data = PCAtraining,method = "gbm",
                 trControl= trainControl(method = "boot632", number = 200),verbose =F)


set.seed(1234)
PCA.lda <- train(Survived ~ ., data = PCAtraining,method = "lda",
                 trControl= trainControl(method = "boot632"), verbose =F)

set.seed(1234)
PCA.rpart <- train(Survived ~ ., data = PCAtraining,method = "rpart",
                   trControl= trainControl(method = "boot632"))


# Let's make a prediction by using final testing set:

PCAtesting <- predict(prePCA,newdata = final_testing)

predictions <- NULL

models <- list(PCA.lda,
               PCA.gbm,
               PCA.rf,
               PCA.amdai,
               PCA.rpart,
               PCA.svm)

predictions <- sapply(models, function(x){
        temp <- as.numeric(as.character(predict(x,newdata = PCAtesting[,-1])))
})

prediction.table <- NULL
for(i in seq_along(predictions)){
        prediction.table <- data.frame(PassengerId = final_testing$PassengerId, 
                                       Survived = predictions[,i])   
        write.csv(prediction.table,paste0("PCA_predictions3_",i,".csv"), row.names = F)
}

# Note of the other models trained in this new data set improved the accuracy, indeed 
# for some models, accuracy becomes worse compared to scratchpad2!

# Still remains as the benchmark:
set.seed(1234)
PCA.svm <- train(Survived ~ ., data = PCAtraining,method = "svmRadial",
                 trControl= trainControl(method = "boot632"), verbose =F)


set.seed(1234)
PCA.svm.cv <- train(Survived ~ ., data = PCAtraining,method = "svmRadial",
                 trControl= trainControl(method = "cv", number = 100), verbose =F)

# Cross validation has some improvement in training accuracy, let's make a prediction:

PCAtesting <- predict(prePCA,newdata = final_testing)


prediction.table <- data.frame(PassengerId = final_testing$PassengerId, 
                               Survived = predict(PCA.svm.cv,PCAtesting))   
write.csv(prediction.table,paste0("PCA_predictions_3_7",".csv"), row.names = F)
# Not improved compared to bootstrap632 model.

# PCA.svm still is the bechmark! 0.794 accuracy.

