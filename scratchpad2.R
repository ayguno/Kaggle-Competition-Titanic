#########################################################
# Trying to perform a fresh look into the problem
# 5/25/2017
# 
#########################################################

library(dplyr);library(ggplot2); library(caret)

training <- read.csv("train.csv")
final_testing <- read.csv("test.csv")


training <- dplyr::select(training, -PassengerId)
training$Survived <- factor(training$Survived)

# Fare is important to keep, Age perhaps not
qplot(x = Age, y = Fare, data = training, color = Survived)

# Also keep Pclass (good at classifiying victims)
qplot(x = Pclass, y = Fare, data = training, color = Survived, alpha = I(0.2))


# Leave Pclass and Parch
F.size = training$Parch + training$SibSp
qplot(x = F.size , y = Fare, data = training, color = Survived, alpha = I(0.4))

# Definitely keep Gender
qplot(x = Sex , y = Fare, data = training, color = Survived, alpha = I(0.4))


# remove other features:

training <- dplyr::select(training, -Name, -Age, -Ticket,-Cabin, - Embarked)
final_testing <- dplyr::select(final_testing, -Name, -Age, -Ticket,-Cabin, - Embarked)

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

# Dimension reduction with the training set

prePCA <- preProcess(training[,-1],method = "pca")
PCAtraining <- predict(prePCA,newdata = training)

qplot(x = PC1, y = PC2, data = PCAtraining, color = Survived, alpha = I(0.3))+theme_bw()+
        scale_color_manual(values = c("red","navy"))

qplot(x = PC2, y = PC3, data = PCAtraining, color = Survived, alpha = I(0.3))+theme_bw()+
        scale_color_manual(values = c("red","navy"))

qplot(x = PC1, y = PC3, data = PCAtraining, color = Survived, alpha = I(0.3))+theme_bw()+
        scale_color_manual(values = c("red","navy"))

# PCA predictors seperate classes pretty well.

# Let's train some classifiers using PCA predictors

set.seed(1234)
PCAknn <- train(Survived ~ PC1 +PC2 +PC3, data = PCAtraining,method = "knn",
                trControl= trainControl(method = "boot632"))

# Takes a long time!
# set.seed(1234)
# PCA.adaboost <- train(Survived ~ ., data = PCAtraining,method = "adaboost",
#                    trControl= trainControl(method = "cv"))

set.seed(1234)
PCA.amdai <- train(Survived ~ ., data = PCAtraining,method = "amdai",
                trControl= trainControl(method = "boot632"))

# # Takes long time!
# set.seed(1234)
# PCA.AdaBag <- train(Survived ~ ., data = PCAtraining,method = "AdaBag",
#                                 trControl= trainControl(method = "boot632"))

# So far gives the best accuracy: (bench mark: 0.83)##############
set.seed(1234)
PCA.rf <- train(Survived ~ PC1 + PC2 + PC3, data = PCAtraining,method = "rf",
                   trControl= trainControl(method = "boot632"))
##################################################################
# Not as good as the PCA.rf
set.seed(1234)
PCA.gbm <- train(Survived ~ ., data = PCAtraining,method = "gbm",
                trControl= trainControl(method = "boot632"), verbose =F)

# Let's make a prediction by using final testing set:

PCAtesting <- predict(prePCA,newdata = final_testing)
predictions <- predict(PCA.rf,PCAtesting[,-1])

prediction.table <- data.frame(PassengerId = final_testing$PassengerId, 
                               Survived = predictions)

write.csv(prediction.table,"PCArf_predictions.csv", row.names = F)

# This prediction has 0.74 test accuracy, similar to earlier feature engineered classifier.

# Not as good as the PCA.rf
set.seed(1234)
PCA.lda <- train(Survived ~ ., data = PCAtraining,method = "lda",
                 trControl= trainControl(method = "boot632"), verbose =F)


# # Takes long time and not better than rf, either.
# set.seed(1234)
# PCA.deepboost <- train(Survived ~ PC1 + PC2 + PC3, data = PCAtraining,method = "deepboost",
#                 trControl= trainControl(method = "boot632"))
# # Takes long time and not better than rf, either.
# set.seed(1234)
# PCA.lssvmLinear <- train(Survived ~ ., data = PCAtraining,method = "lssvmRadial",
#                        trControl= trainControl(method = "cv"))
