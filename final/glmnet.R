
### Regression Analysis
### Quantitative Psychology 1


## Data Reading

# install.packages("car")
library(foreign, pos = 14)
library(graphics)
library(Hmisc)
library(ggplot2)
library(glmnet)
library(car)

labels <- read.csv("scale_labels.csv")
f <- na.omit(read.csv("results_compilation.csv"))
str(f)

data <- f[27:64]
drops <- c("RFQ", "ERQ", "LOC13", "SMS", "BPSSR_I5", "BPSSR_Amb", "BPSSR_I", "BPSSR_E")
data <- data[ , !(names(data) %in% drops)]

obs <- which(colnames(data)=="FS")
x <- as.matrix(data[, -obs])
y <- data[, obs]

## glmnet

# ridge: reduce the coefficient of all parameters
# lasso: make some of the coefficients zero
# both ridge and lasso regression techniques are used to reduce regression coefficients to simplify the models
# when using glmnet, glmnet becomes ridge when alpha=1 and lasso when alpha=0


## Ridge Regression

# regression coefficient changes according to the size of lambda
# lambda determines the degree of penalty applied to large coefficients
# if the regression coefficient is too large, the risk of overfitting increases
# cross-validation is used to prevent overfitting

fit1 <- glmnet(x, y, family="gaussian", alpha = 0)
cv1 <- cv.glmnet(x, y, family="gaussian", alpha = 0)

cv1$lambda.min # value when cross validation is highest
coef(cv1, s = "lambda.min")

plot(fit1, xvar = "lambda")
plot(cv1)

ridge.fit <- glmnet(x, y, family="gaussian", alpha = 0, lambda = cv1$lambda.min)
coef(ridge.fit)

ridge.pred <- predict(ridge.fit, s = cv1$lambda.min, type = 'coefficients')


## Lasso Regression

fit2 <- glmnet(x, y, family = "gaussian", alpha = 1)
cv2 <- cv.glmnet(x, y, family = "gaussian", alpha = 1)
cv2$lambda.min

plot(fit2, xvar = "norm")
plot(cv2) # as lambda increases, regression coefficient decreases and mean-squared error decreases

lasso.fit <- glmnet(x, y, family="gaussian", alpha = 1, lambda = cv2$lambda.min)
coef(lasso.fit)

lasso.pred <- predict(lasso.fit, s = cv2$lambda.min, type = 'coefficients')

