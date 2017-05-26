#########################################################
# Trying to perform a fresh look into the problem
# 5/25/2017
# 
#########################################################

library(dplyr);library(ggplot2)

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

lmFit1 <- lm(Fare ~ factor(Pclass), data = training)