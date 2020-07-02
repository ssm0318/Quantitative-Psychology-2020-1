
update.packages(ask = FALSE)

## libraries
install.packages("ggcorrplot", dependencies = TRUE)
install.packages("corrplot", dependencies = TRUE)
install.packages("ggplot2", dependencies = TRUE)
install.packages("gridExtra", dependencies = TRUE)
install.packages("irr", dependencies = TRUE)
install.packages("dplyr", dependencies = TRUE)
install.packages("psych", dependencies = TRUE)

library(ggcorrplot)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(irr)
require(gridExtra)
library(dplyr)
library(psych)

## pre-process data

# read files
raw_data <- read.csv("../raw_data.csv")
head(raw_data)

agg_data <- read.csv("../results_compilation.csv")
head(agg_data)

labels <- read.csv("../scale_labels.csv", header = TRUE)

data_complete <- agg_data[agg_data$Complete == "TRUE",]

self_eval <- data_complete[data_complete$SurveyType == 1,]
others_eval <- data_complete[data_complete$SurveyType == 2,]

attach(data_complete)

# scales and demographics

demographics <- data_complete[, 1:26]
self_demographics <- self_eval[, 1:26]
others_demographics <- others_eval[, 1:26]

scales <- data_complete[, 27:64]
self_scales <- self_eval[, 27:64]
others_scales <- others_eval[, 27:64]

head(scales)
head(self_scales)
head(others_scales)

##########################
##########################

## descriptive

# 36 self-eval, 34 others-eval

head(demographics)

# gender
table(demographics$Gender)
# F: 53, M: 16, UND: 1

# duration
boxplot(demographics$DurationTotal)
summary(demographics$DurationTotal)

# guess
others_demographics$guess_total <- 
  rowMeans(others_demographics[c('guess_1', 'guess_2', 'guess_3')])
summary(others_demographics$guess_total)
boxplot(others_demographics$guess_total)

boxplot(FS)

##########################
##########################

## correlation

corr <- cor(scales, use = "complete.obs")
self_corr <- cor(self_scales, use = "pairwise.complete.obs")
others_corr <- cor(others_scales, use = "complete.obs")
head(corr[, 1:5])
head(self_corr[, 1:5])
head(others_corr[, 1:5])

# matrix of p-values 
p.mat <- cor_pmat(scales)
self_p.mat <- cor_pmat(self_scales)
others_p.mat <- cor_pmat(others_scales)
head(p.mat[,1:4])

# correlation matrix 1 - heat map style
ggcorrplot(corr)
plot1 <- ggcorrplot(self_corr, show.legend = FALSE)
plot2 <- ggcorrplot(others_corr, show.legend = FALSE)
grid.arrange(plot1, plot2, ncol = 2)

# correlation matrix 2 - differing size
cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

# correlation matrix 3
corrplot(corr, type="upper", order="hclust", p.mat = p.mat, sig.level = 0.05, insig = "blank", diag = FALSE)

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(corr, method="color", col=col(200),  
         type="upper", order="hclust", 
         # Combine with significance
         p.mat = p.mat, sig.level = 0.05, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag=FALSE 
)


##########################
##########################

## intra class correlation

# corr matrix as list
corr_list <- as.data.frame(as.table(corr))
combinations = combn( colnames( corr ) , 2 , FUN = function( x ) { paste( x , collapse = "x" ) } )
corr_list = corr_list[ corr_list$Var1 != corr_list$Var2 , ]
corr_list = corr_list[ paste( corr_list$Var1 , corr_list$Var2 , sep = "x" ) %in% combinations , ]

self_corr_list <- as.data.frame(as.table(self_corr))
combinations = combn( colnames( self_corr ) , 2 , FUN = function( x ) { paste( x , collapse = "x" ) } )
self_corr_list = self_corr_list[ self_corr_list$Var1 != self_corr_list$Var2 , ]
self_corr_list = self_corr_list[ paste( self_corr_list$Var1 , self_corr_list$Var2 , sep = "x" ) %in% combinations , ]

others_corr_list <- as.data.frame(as.table(others_corr))
combinations = combn( colnames( others_corr ) , 2 , FUN = function( x ) { paste( x , collapse = "x" ) } )
others_corr_list = others_corr_list[ others_corr_list$Var1 != others_corr_list$Var2 , ]
others_corr_list = others_corr_list[ paste( others_corr_list$Var1 , others_corr_list$Var2 , sep = "x" ) %in% combinations , ]

# ICC
agg_corr <- cbind(self_corr_list[,3], others_corr_list[,3])
icc <- icc(agg_corr, model = "twoway", type = "consistency")
icc$

##########################
##########################

## correlation pairs

# boredom proneness ~ ADHD : 0.346 (.00681)
corr1 <- cor.test(ADHD_RS, BPS_SR)
corr1

# external BP ~ ADHD : 0.487 (7.92e-05)
corr2 <- cor.test(ADHD_RS, BPSSR_E)
corr2

# internal BP ~ ADHD : 0.180 (0.168); NOT significant
# same for BPSSR_I, BPSSR_Amb
corr3 <- cor.test(ADHD_RS, BPSSR_I)
corr3

# ADHD ~ risk propensity : 0.415 (0.000877)
corr4 <- cor.test(ADHD_RS, RPS)
corr4

# ADHD ~ boredom susceptibility : 0.377 (0.00300)
corr5 <- cor.test(ADHD_RS, ZBS)
corr5

# ADHD ~ procrastination : 0.590 (5.53e-07)
corr6 <- cor.test(ADHD_RS, PCS_9)
corr6

# ADHD ~ emotional regulation : -0.289 (0.0239)
# subscales were NOT significant
corr7 <- cor.test(ADHD_RS, ERQ_10)
corr7

# ADHD ~ flourishing : -0.361 (0.00429)
corr8 <- cor.test(ADHD_RS, FS)
corr8

# ADHD ~ aggression : 0.497 (4.663e-05) 
corr9 <- cor.test(ADHD_RS, BPAQ_SF)
corr9

# ADHD ~ prevention focus : -0.557 (3.119e-06)
corr10 <- cor.test(ADHD_RS, RFQ_v)
corr10

# ADHD ~ promotion focus : 0.329 (0.00964)
corr11 <- cor.test(ADHD_RS, RFQ_m)
corr11

# ADHD ~ locus of control : 0.260 (0.0446)
# higher LOC_13 score implies external locus of control
corr12 <- cor.test(ADHD_RS, LOC_13)
corr12

# ADHD ~ cognitive/affective mindfulness : -0.576 (1.487e-06)
corr13 <- cor.test(ADHD_RS, CAMS_R)
corr13

# ADHD ~ boredom susceptibility : 0.377 (0.00299)
corr14 <- cor.test(ADHD_RS, ZBS)
corr14

# ADHD ~ internet addiction : 0.496 (4.86e-05)
corr15 <- cor.test(ADHD_RS, SAS)
corr15

corr <- cor.test(ADHD_RS, SOAS)
corr

corr_E <- cor.test(BPSSR_E, ERQ_10_e)
corr_I <- cor.test(BPSSR_I, ERQ_10_e)
corr_E
corr_I
