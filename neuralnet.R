install.packages("neuralnet", dependencies = TRUE)
install.packages("ade4", dependencies = TRUE)
install.packages("mlbench", dependencies = TRUE)
install.packages("kohonen", dependencies = TRUE)

library(neuralnet)
library(ade4)
library(mlbench)
library(kohonen)


## Backpropagation

# data generation
A<-c(0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1)
B<-c(1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,0)
C<-c(0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,1,1,0)
D<-c(1,1,1,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,1,1,0)
E<-c(1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1)
F<-c(1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0)
G<-c(0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,1,1,1,0)
H<-c(1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1)
I<-c(0,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0)
J<-c(1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,1,1,0,0)
K<-c(1,0,0,0,1,1,0,0,1,0,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,0,1)
L<-c(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,1,1)
M<-c(1,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1)
N<-c(1,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0,0,1)
O<-c(0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0)
P<-c(1,1,1,1,0,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0)
Q<-c(0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,1,0,1,1,0,0,1,0,0,1,1,0,1)
R<-c(1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,1,0,0,1,0,0,1,0,1,0,0,0,1)
S<-c(0,1,1,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,1,1,0)
T<-c(1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0)
U<-c(1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0)
V<-c(1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0)
W<-c(1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,1,0,1,1,1,0,1,1,1,0,0,0,1)
X<-c(1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,0,1)
Y<-c(1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0)
Z<-c(1,1,1,1,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,1,1,1)

# draw given array (i.e. alphabet)
dlet<-function(A) {
  tt<-t(array(A,dim=c(5,7)))
  table.value(tt, csi = 2, clabel.r = 2, clabel.c = 2,clegend=0)
  par(mfrow = c(1, 1))
}

# draw R
dlet(R)

# three layer perceptron
input <- rbind(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z)
target <- c(1:26) # assign 1 to A, 2 to B, 3 to C, ...
train <- data.frame(input, target)
n <- names(train)

# train using neural network; 3 nodes used in hidden layer
f <- as.formula(paste('target ~', paste(n[!n %in% 'target'], collapse = ' + ')))
nn1 <- neuralnet(f, data = train, hidden = 3, rep = 5, err.fct = "sse", linear.output = TRUE)
plot(nn1)

res1 <- compute(nn1, input)
print(round(res1$net.result)) # test the model w/ 3 nodes; some errors are found

# neural network trained with 9 nodes in hidden layer
nn2 <- neuralnet(f, data = train, hidden = 9, rep = 5, err.fct = "sse", linear.output = TRUE)
res2 <- compute(nn2, input)
print(round(res2$net.result)) # better!


## Kohonen network

# data; use data1 and data2 from Machine learning benchmark provided by R
data1 = mlbench.cassini(200)
plot(data1)
head(data1$x)

head(data1$classes)

data2 = mlbench.spirals(200, 1.5, 0.05)
plot(data2)

data1 = as.data.frame(data1)
data2 = as.data.frame(data2)

par(mfrow = c(1, 2), pty = "s") # make square axes
plot(data1$x.1, data1$x.2, col = data1$classes, main = 'data1')
plot(data2$x.1, data2$x.2, col = data2$classes, main = 'data2')

# SOM (Self-Organizing Maps) Analysis

mygrid = somgrid(2, 3, "hexagonal") # defined grid (topology) for SOM as hexagonal
plot(mygrid$pts[, 1], mygrid$pts[, 2]) # a peek at what the grid looks like

s1 = som(as.matrix(data1[, 1:2]), grid = mygrid, rlen = 100)
plot(data1$x.1, data1$x.2, col = s1$unit.classif, main = 'SOM classification')
plot(s1)

s2 = som(as.matrix(data2[, 1:2]), grid = somgrid(10, 10, "hexagonal"), rlen = 100)
plot(data2$x.1, data2$x.2, col = s2$unit.classif, main = 'SOM classification')
plot(s2)


## analysis using wine data

# separate training and testing data
data(wines)
set.seed(7)
training <- sample(nrow(wines), 120)
Xtraining <- scale(wines[training, ])
Xtest <- scale(wines[-training, ], center = attr(Xtraining, "scaled:center"),
               scale = attr(Xtraining, "scaled:scale"))

# The SOM model
som.wines <- som(Xtraining, grid = somgrid(5, 5, "hexagonal"))
plot(som.wines, type = "codes")
som.prediction <- predict(som.wines, newdata = Xtest,
                          trainX = Xtraining, trainY = factor(wine.classes[training]))
som.prediction$prediction
