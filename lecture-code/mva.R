install.packages("factoextra", dependencies = TRUE)

library(graphics)
library(MASS)
library(psych)
library(cluster)
library(factoextra)
library(cluster)

data("USArrests")
head(USArrests)

USArrests <- scale(USArrests)
dim(USArrests)


## Hierarchical Cluster Analysis
dist.mat <- as.dist(1-cor(t(USArrests)))

h <- hclust(dist.mat, method = "ward.D")
plot(h)
clusterCut <- cutree(h, k = 3)
rect.hclust(h, k = 3)


## K-means Cluster Analysis

# find the optimal number of clusters
t = fviz_nbclust(USArrests, kmeans, method = 'wss')
t + geom_vline(xintercept = 4, linetype = 5, col = "darkred")

km.res <- kmeans(USArrests, 3, nstart = 20)
km.res
km.res$cluster # in which cluster each belongs to

df_member <- cbind(USArrests, cluster = km.res$cluster)
head(df_member, 10)

fviz_cluster(km.res, data = USArrests,
             palette = c("red", "blue", "black", "darkgreen"),
             ellipse.type = "euclid",
             star.plot = T,
             repel = T,
             ggtheme = theme())


## Principal Component Analysis
online <- read.csv("OnlineNewsPopularity.csv")
x <- as.matrix(online[, c(3:60)])
x2 <- apply(x, 2, scale, scale = TRUE)
d <- dist(t(x2))

pca <- prcomp(x2, retx = TRUE, scale = TRUE)
summary(pca)
fviz_eig(pca)

fviz_pca_var(pca, axes = c(1, 2),
             col.var = "contrib",
             gradient.cols = c("#00AFBB", "E7B800", "FC4E07"),
             repel = TRUE)

fviz_pca_var(pca, axes = c(2, 3),
             col.var = "contrib",
             gradient.cols = c("#00AFBB", "E7B800", "FC4E07"),
             repel = TRUE)

varimax3 <- varimax(pca$rotation[, 1:3])
print(varimax3)

promax3 <- promax(pca$rotation[, 1:3])
print(promax3)


## Multidimensional Scaling
data("eurodist")
d <- dist(eurodist)
loc <- cmdscale(d)
plot(loc, type = "n")
text(loc[, 1], loc[, 2], labels(d))

plot(loc[, 2], -1 * loc[, 1], type = "n")
text(loc[, 2], -1 * loc[, 1], labels(d))

d <- dist(t(x2))
loc <- isoMDS(d, k = 2, p = 1)

x <- loc$points[, 1]
y <- loc$points[, 2]

plot(loc$points, type = "n")
text(x, y, names(x), cex = 1)


