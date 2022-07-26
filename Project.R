library("GGally")
library("ggplot2")
library("caret")
library("glmnet")
library("gridExtra")
library("lars")

data <- read.csv("student_7327data.csv")

#exploratory data analysis

is.na(data)

summary(data)

dev.off()

#plot the data
ggpairs(data,
        upper=list(continuous=wrap("points"), alpha=0.4),
        lower="blank", axisLabels="none")

p1 <- ggplot(data, aes(x=Var1, y=Var9)) + 
  geom_point()

p2 <- ggplot(data, aes(x=Var1, y=Var5)) + 
  geom_point()

p3 <- ggplot(data, aes(x=Var31, y=Var5)) + 
  geom_point()

p4 <- ggplot(data, aes(x=Var6, y=Var9)) + 
  geom_point()

grid.arrange(p1, p2, p3, p4)

#check correlation
cor(data)

str(data)



#split data
sample_size <- floor(0.7 * nrow(data))
training_index <- sample(seq_len(nrow(data)), size = sample_size)
data_train <- data[training_index, ]
data_test <- data[-training_index, ]

#standardise and center data

scalingData <- preProcess(data_train[,-32],method = c("center", "scale"))
data_train[,-32] <- predict(scalingData,data_train[,-32])
data_test[,-32] <- predict(scalingData,data_test[,-32])

#ridge regression

x <- as.matrix(data_train[,1:31])
y <- data_train$class


set.seed(37)
fit.ridge <- glmnet(x,y, alpha=0, family = "binomial")
plot(fit.ridge,xvar="lambda",label=TRUE)

cv.ridge <- cv.glmnet(x,y,alpha=0, family="binomial")
plot(cv.ridge)

cv.ridge$lambda.1se

cv.ridge$lambda.min

fit.ridge.lambda.min <- glmnet(x, y, alpha = 0, lambda=cv.ridge$lambda.1se, family="binomial")

coef(fit.ridge.lambda.min)

#prediction for ridge regression

pred.ridge <- fit.ridge.lambda.min %>% predict(as.matrix(data_test[, -32]))
predicted.classes.ridge <- ifelse(pred.ridge > 0.5, 1, 0)

#calculate classification rate
mean(predicted.classes.ridge == data_test$class)


#lasso regression

set.seed(37)
fit.lasso <- glmnet(x,y,alpha=1, family = "binomial")
plot(fit.lasso,xvar="lambda",label=TRUE)

cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(cv.lasso)

cv.lasso$lambda.1se

cv.lasso$lambda.min

lasso.model <- glmnet(x, y, alpha = 1, family = "binomial",
                      lambda = cv.lasso$lambda.min)

coef(lasso.model)

#predictions

pred.lasso <- lasso.model %>% predict(as.matrix(data_test[, -32]))

predicted.classes.lasso <- ifelse(pred.lasso > 0.5, 1, 0)

#calculate classification rate

mean(predicted.classes.lasso == data_test$class)



#elastic net

set.seed(37)
model.net <- train(
  as.factor(class)~.-1, data = data_train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10)

model.net$bestTune

coef(model.net$finalModel, model.net$bestTune$lambda)

#predictions

predictions.net <- model.net %>% predict(as.matrix(data_test[, -32]))

#calculate classification rate

mean(predictions.net == data_test$class)

tab.pred<-table(data_test$class, predictions.net)
tab.pred

tab.rate<-sweep(tab.pred,1,apply(tab.pred,1,sum),"/")
tab.rate

