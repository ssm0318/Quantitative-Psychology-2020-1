# Read in files (provide a full path, e.g., "~/Desktop/sample_dataset/users-likes.csv")
users <- read.csv("users.csv")
likes <- read.csv("likes.csv")
ul <- read.csv("users-likes.csv")    

# You can check what's inside each object using the following set of commands:
head(users)
head(likes)
head(ul)

tail(ul)
tail(users)
tail(likes)

dim(ul)
dim(users)
dim(likes)

# Match entries in ul with users and likes dictionaries
ul$user_row<-match(ul$userid,users$userid)
ul$like_row<-match(ul$likeid,likes$likeid)

# and inspect what happened: 
head(ul)

# Install Matrix library - run only once
# install.packages("Matrix")

# Load Matrix library
require(Matrix)

# Construct the sparse User-Like Matrix M
M <- sparseMatrix(i = ul$user_row, j = ul$like_row, x = 1)

# Check the dimensions of M
dim(M)

# Save user IDs as row names in M
rownames(M) <- users$userid

# Save Like names as column names in M
colnames(M) <- likes$name

# Remove ul and likes objects (they won't be needed)
rm(ul, likes)

# Remove users/Likes occurring less than 50/150 times

repeat {                                       # Repeat whatever is in the brackets
  i <- sum(dim(M))                             # Check the size of M
  M <- M[rowSums(M) >= 50, colSums(M) >= 150]  # Retain only these rows/columns that meet the threshold
  if (sum(dim(M)) == i) break                  # If the size has not changed, break the loop
}

# Check the new size of M
dim(M)

# Remove the users from users object that were removed
# from M
users <- users[match(rownames(M), users$userid), ]

# Check the new size of users
dim(users)

# Preset the random number generator in R 
# for the comparability of the results
set.seed(seed = 68)

# Install irlba package (run only once)
# install.packages("irlba")

# Load irlba and extract 5 SVD dimensions
library(irlba)
Msvd <- irlba(M, nv = 5)

# User SVD scores are here:
u <- Msvd$u

# Like SVD scores are here:
v <- Msvd$v

# The scree plot of singular values:
plot(Msvd$d)

# First obtain rotated V matrix:
# (unclass function has to be used to save it as an 
# object of type matrix and not loadings)
v_rot <- unclass(varimax(Msvd$v)$loadings)

# The cross-product of M and v_rot gives u_rot:
u_rot <- as.matrix(M %*% v_rot)

# Install topicmodels package (run only once)
# install.packages("topicmodels")

# Load it
library(topicmodels)

# Conduct LDA analysis, see text for details on setting
# alpha and delta parameters. 
# WARNING: this may take quite some time!

Mlda <- LDA(M, control = list(alpha = 10, delta = .1, seed=68), k = 5, method = "Gibbs")

# Extract user LDA cluster memberships
gamma <- Mlda@gamma

# Extract Like LDA clusters memberships
# betas are stored as logarithms, 
# function exp() is used to convert logs to probabilities
beta <- exp(Mlda@beta) 

# Log-likelihood of the model is stored here:
Mlda@loglikelihood

# and can be also accessed using logLik() function:
logLik(Mlda)

# Let us estimate the log-likelihood for 2,3,4, and 5 cluster solutions: 
lg <- list()
for (i in 2:5) {
  Mlda <- LDA(M, k = i, control = list(alpha = 10, delta = .1, seed = 68), method = "Gibbs")
  lg[[i]] <- logLik(Mlda) 
}

plot(2:5, unlist(lg))   

# Correlate user traits and their SVD scores
# users[,-1] is used to exclude the column with IDs
cor(u_rot, users[,-1], use = "pairwise")

# LDA version
cor(gamma, users[,-1], use = "pairwise")

# You need to install ggplot2 and reshape2 packages first, run only once:
install.packages("ggplot2", "reshape2")

# Load these libraries
library(ggplot2)
library(reshape2)

# Get correlations
x<-round(cor(u_rot, users[,-1], use="p"),2)

# Reshape it in an easy way using ggplot2
y<-melt(x)
colnames(y)<-c("SVD", "Trait", "r")

# Produce the plot
qplot(x=SVD, y=Trait, data=y, fill=r, geom="tile") +
  scale_fill_gradient2(limits=range(x), breaks=c(min(x), 0, max(x)))+
  theme(axis.text=element_text(size=12), 
        axis.title=element_text(size=14,face="bold"),
        panel.background = element_rect(fill='white', colour='white'))+
  labs(x=expression('SVD'[rot]), y=NULL)

# SVD
top <- list()
bottom <-list()
for (i in 1:5) {
  f <- order(v_rot[ ,i])
  temp <- tail(f, n = 10)
  top[[i]]<-colnames(M)[temp]  
  temp <- head(f, n = 10)
  bottom[[i]]<-colnames(M)[temp]  
}

# LDA
top <- list()
for (i in 1:5) {
  f <- order(beta[i,])
  temp <- tail(f, n = 10)
  top[[i]]<-colnames(M)[temp]  
}

# Split users into 10 groups
folds <- sample(1:10, size = nrow(users), replace = T)

# Take users from group 1 and assign them to the TEST subset
test <- folds == 1

# Extract SVD dimensions from the TRAINING subset
# training set can be accessed using !test
Msvd <- irlba(M[!test, ], nv = 50)

# Rotate Like SVD scores (V)
v_rot <- unclass(varimax(Msvd$v)$loadings)

# Rotate user SVD scores *for the entire sample*
u_rot <- as.data.frame(as.matrix(M %*% v_rot))

# Build linear regression model for openness
# using TRAINING subset
fit_o <- glm(users$ope~., data = u_rot, subset = !test)

# Inspect the regression coefficients
coef(fit_o)

# Do the same for gender
# use family = "binomial" for logistic regression model
fit_g <- glm(users$gender~.,data = u_rot, subset = !test, family = "binomial")

# Compute the predictions for the TEST subset
pred_o <- predict(fit_o, u_rot[test, ])
pred_g <- predict(fit_g, u_rot[test, ], type = "response")

# Correlate predicted and actual values for the TEST subset
r <- cor(users$ope[test], pred_o)
r

# Compute Area Under the Curve for gender
# remember to install ROCR library first
install.packages("ROCR")
library(ROCR)
temp <- prediction(pred_g, users$gender[test])
auc <- performance(temp,"auc")@y.values
auc

# Choose which k are to be included in the analysis
ks<-c(2:10,15,20,30,40,50)

# Preset an empty list to hold the results
rs <- list()

# Run the code below for each k in ks
for (k in ks){
  # Varimax rotate Like SVD dimensions 1 to k
  v_rot <- unclass(varimax(Msvd$v[, 1:k])$loadings)
  
  # This code is exactly like the one discussed earlier
  u_rot <- as.data.frame(as.matrix(M %*% v_rot))
  fit_o <- glm(users$ope~., data = u_rot, subset = !test)
  pred_o <- predict(fit_o, u_rot[test, ])
  
  # Save the resulting correlation coefficient as the
  # element of R called k
  rs[[as.character(k)]] <- cor(users$ope[test], pred_o)
}

# Check the results
rs

# Convert rs into the correct format
data<-data.frame(k=ks, r=as.numeric(rs))

# plot!
ggplot(data=data, aes(x=k, y=r, group=1)) + 
  theme_light() +
  stat_smooth(colour="red", linetype="dashed", size=1,se=F) + 
  geom_point(colour="red", size=2, shape=21, fill="white") +
  scale_y_continuous(breaks = seq(0, .5, by = 0.05))

pred_o <- rep(NA, n = nrow(users))
for (i in 1:10){
  test <- folds == i
  Msvd <- irlba(M[!test, ], nv = 50)
  v_rot <- unclass(varimax(Msvd$v)$loadings)
  u_rot <- as.data.frame(as.matrix(M %*% v_rot))
  fit_o <- glm(users$ope~., data = u_rot, subset = !test)
  pred_o[test] <- predict(fit_o, u_rot[test, ])
}
r <- cor(users$ope, pred_o)

Mlda <- LDA(M[!test, ], control=list(alpha=1, delta=.1, seed=68), k=50, method="Gibbs")
temp<-posterior(Mlda, M)
gamma<-as.data.frame(temp$topics)

# Load libraries
require(Matrix)
library(ROCR)
library(topicmodels)
library(irlba)

# Load files
users<-read.csv("users.csv")
likes<-read.csv("likes.csv")
ul<-read.csv("users-likes.csv")

# Construct the matrix
ul$user_row<-match(ul$userid,users$userid)
ul$like_row<-match(ul$likeid,likes$likeid)

M<-sparseMatrix(i=ul$user_row,j=ul$like_row,x=1)
rownames(M)<-users$userid
colnames(M)<-likes$name
rm(ul,likes)

# Matrix trimming
while (T){
  i<-sum(dim(M))
  M<-M[rowSums(M)>=50, colSums(M)>=150]
  if (sum(dim(M))==i) break
}
users <- users[match(rownames(M),users$userid), ]

# Start predictions
set.seed(seed=68)
n_folds<-10                # set number of folds
k<-50                      # set k
vars<-colnames(users)[-1]  # choose variables to predict

folds <- sample(1:n_folds, size = nrow(users), replace = T)

results<-list()
for (fold in 1:n_folds){ 
  print(paste("Cross-validated predictions, fold:", fold))
  test <- folds == fold
  
  # If you want to use SVD:
  Msvd <- irlba(M[!test, ], nv = k)
  v_rot <- unclass(varimax(Msvd$v)$loadings)
  predictors <- as.data.frame(as.matrix(M %*% v_rot))
  
  # If you want to use LDA, comment out the SVD lines above, and uncomment two lines below
  # Mlda <- LDA(M[!test, ], control = list(alpha = 1, delta = .1, seed=68), k = k, method = "Gibbs")
  # predictors <- as.data.frame(posterior(Mlda,M, control = list(alpha = 1, delta = .1))$topics)
  
  for (var in vars){
    results[[var]]<-rep(NA, n = nrow(users))
    # check if the variable is dichotomous
    if (length(unique(na.omit(users[,var]))) ==2) {    
      fit <- glm(users[,var]~., data = predictors, subset = !test, family = "binomial")
      results[[var]][test] <- predict(fit, predictors[test, ], type = "response")
    } else {
      fit<-glm(users[,var]~., data = predictors, subset = !test)
      results[[var]][test] <- predict(fit, predictors[test, ])
    }
    print(paste(" Variable", var, "done."))
  }
}

compute_accuracy <- function(ground_truth, predicted){
  if (length(unique(na.omit(ground_truth))) ==2) {
    f<-which(!is.na(ground_truth))
    temp <- prediction(predicted[f], ground_truth[f])
    return(performance(temp,"auc")@y.values)
  } else {return(cor(ground_truth, predicted,use = "pairwise"))}
}

accuracies<-list()
for (var in vars) accuracies[[var]]<-compute_accuracy(users[,var], results[[var]])