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

